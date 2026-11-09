package catanai.stac;

import java.util.function.Consumer;

import com.google.gson.JsonArray;
import com.google.gson.JsonElement;
import com.google.gson.JsonObject;

import catanai.jsettlers.DecisionLog;
import catanai.jsettlers.PythonBridge;

import soc.dialogue.StacDialogueManager;
import soc.dialogue.StacTradeMessage;
import soc.game.SOCCity;
import soc.game.SOCDevCardConstants;
import soc.game.SOCGame;
import soc.game.SOCPlayer;
import soc.game.SOCPlayingPiece;
import soc.game.SOCResourceSet;
import soc.game.SOCRoad;
import soc.game.SOCSettlement;
import soc.game.SOCTradeOffer;
import soc.game.StacTradeOffer;
import soc.message.SOCGameTextMsg;
import soc.message.SOCGameState;
import soc.robot.SOCRobotBrainImpl;
import soc.robot.SOCRobotClient;
import soc.robot.stac.Persuasion;
import soc.robot.stac.StacRobotBrain;
import soc.util.CappedQueue;
import soc.util.SOCRobotParameters;

/**
 * StacSettlers (JSettlers 1.1.16 based) robot brain whose every decision is asked of the Catan AI
 * served policy through {@link PythonBridge} (the same Python bridge as the JSettlers2 robot).
 * STAC keeps tracking the game and stays authoritative; the brain only sends the chosen action
 * through the normal client API.
 *<P>
 * Decisions bridged: initial settlements and roads, roll vs. playing a development card before the
 * roll, each PLAY1 action (build, buy, bank/port trade, play a card, end turn), free roads from Road
 * Building, robber placement and victim, and discards.
 *<P>
 * Player trading has two match modes. With STAC's chat negotiation off (the default), the match
 * patch blocks player exchanges, the bridge disables the policy's offers, and an offer to us is left
 * to STAC and logged as a fallback. With chat negotiation on ({@link StacRobotBrain#isChatNegotiation()},
 * set by the Simulation config), trades use STAC's chat format ({@link StacTradeMessage}): an offer
 * {@code TRD:from:to:<offer>}, answers {@code ACC}, {@code REJ} or {@code NO_RESPONSE}, and the
 * offerer's confirming {@code ACC}, after which STAC's server executes the exchange. This brain then
 * <ul>
 * <li>answers every chat offer addressed to it with the policy's accept/reject decision (the same
 *     {@code trade_response} request as the JSettlers bridge), or a reject when the offer cannot be a
 *     legal trade (not involving the current player, or a give set we do not hold);</li>
 * <li>sends the policy's own {@code offer} as a chat offer, waits for the recipient's answer, and
 *     confirms an accept so the server executes it (a reject, no-response or counter-offer ends the
 *     attempt and the policy is asked again);</li>
 * <li>counts each player's public chat answers to the current player's offers for the snapshot's
 *     {@code trade_answers}, as the JSettlers client does for protocol answers.</li>
 * </ul>
 * Every chat trade event is written to the decision log as a {@code chat_trade} record.
 *<P>
 * When the bridge cannot give a STAC-legal action, the inherited STAC/JSettlers 1.1.16 logic
 * ({@link SOCRobotBrainImpl}) decides and the decision is logged as a {@code fallback} with its
 * reason. Bridge transport/exception failures are logged as decisions with {@code cmd=error} (bridge
 * errors) and also fall back so the game continues.
 */
public class CatanAiStacBrain extends SOCRobotBrainImpl
{
    /** Hard cap on bridged PLAY1 decisions in one turn, against request loops. */
    static final int MAX_DECISIONS_PER_TURN = 80;
    /** Identical consecutive PLAY1 commands allowed before treating the command as refused. */
    static final int MAX_REPEATS = 4;

    private boolean started;
    private boolean ended;
    private boolean preRollCardPending;
    private boolean freeRoadsBeforeRoll;
    private int decisionsThisTurn;
    private int turnSeen = -1;
    private String lastCommand;
    private int repeats;
    private int ourPN = -1;
    private String gameName;

    /** Per player number: {accepted, answered} public chat answers to the current player's offers. */
    private final int[][] chatAnswers = new int[4][2];
    /** Our chat offer awaiting its recipient's answer, or null. */
    private StacTradeOffer pendingOffer;
    private int pendingTo = -1;

    public CatanAiStacBrain(final SOCRobotClient rc, final SOCRobotParameters params, final SOCGame ga,
        final CappedQueue mq)
    {
        super(rc, params, ga, mq);
        gameName = ga.getName();
    }

    // ------------------------------------------------------------------ bridge calls

    static String cmd(final JsonObject r)
    {
        final JsonElement c = r.get("cmd");
        return (c == null || c.isJsonNull()) ? "error" : c.getAsString();
    }

    private void logStart()
    {
        if (started)
            return;
        started = true;
        ourPN = ourPlayerData.getPlayerNumber();
        final JsonObject rec = new JsonObject();
        rec.addProperty("type", "game_start");
        rec.addProperty("game", game.getName());
        rec.addProperty("seat", ourPN);
        rec.addProperty("first_player", game.getFirstPlayer());
        rec.addProperty("bot", client.getNickname());
        final JsonArray names = new JsonArray();
        for (int pn = 0; pn < game.maxPlayers; ++pn)
            names.add(game.getPlayer(pn).getName());
        rec.add("players", names);
        DecisionLog.write(rec);
    }

    JsonObject ask(final String decision, final Consumer<JsonObject> extra)
    {
        logStart();
        final int pn = ourPlayerData.getPlayerNumber();
        final long t0 = System.nanoTime();
        JsonObject reply;
        try
        {
            final JsonObject snap = StacStateJson.snapshot(game, pn, decision);
            snap.addProperty("kind", "decide");
            snap.add("js_legal", StacStateJson.legal(game, pn, decision));
            // Public chat answers to offers (all zero when player trading is off).
            final JsonArray answers = new JsonArray();
            synchronized (chatAnswers)
            {
                for (int i = 0; i < game.maxPlayers; ++i)
                {
                    final JsonArray x = new JsonArray();
                    x.add(chatAnswers[i][0]);
                    x.add(chatAnswers[i][1]);
                    answers.add(x);
                }
            }
            snap.add("trade_answers", answers);
            if (extra != null)
                extra.accept(snap);
            reply = PythonBridge.get().call(snap);
        } catch (Exception e) {
            reply = new JsonObject();
            reply.addProperty("cmd", "error");
            reply.addProperty("error", "java: " + e);
        }
        final double ms = (System.nanoTime() - t0) / 1e6;
        final JsonObject rec = new JsonObject();
        rec.addProperty("type", "decision");
        rec.addProperty("game", game.getName());
        rec.addProperty("seat", pn);
        rec.addProperty("decision", decision);
        rec.addProperty("turn_count", game.getTurnCount());
        rec.addProperty("cmd", cmd(reply));
        rec.addProperty("latency_ms", Math.round(ms * 100) / 100.0);
        for (final String k : new String[] { "think_ms", "reason", "error", "legal_mismatches", "action", "suppressed_offer" })
            if (reply.has(k))
                rec.add(k, reply.get(k));
        DecisionLog.write(rec);
        return reply;
    }

    /** Record that STAC's own logic made this decision instead of the policy. */
    void fallback(final String decision, final JsonObject reply)
    {
        final JsonObject rec = new JsonObject();
        rec.addProperty("type", "fallback");
        rec.addProperty("game", game.getName());
        rec.addProperty("seat", ourPlayerData.getPlayerNumber());
        rec.addProperty("decision", decision);
        rec.addProperty("turn_count", game.getTurnCount());
        rec.addProperty("cmd", cmd(reply));
        rec.addProperty("reason", reply.has("reason") ? reply.get("reason").getAsString()
            : (reply.has("error") ? "bridge error: " + reply.get("error").getAsString() : "unhandled cmd " + cmd(reply)));
        if (reply.has("action"))
            rec.add("action", reply.get("action"));
        DecisionLog.write(rec);
    }

    private static JsonObject refusal(final String reason)
    {
        final JsonObject r = new JsonObject();
        r.addProperty("cmd", "fallback");
        r.addProperty("reason", reason);
        return r;
    }

    // ------------------------------------------------------------------ initial placement

    @Override
    protected void placeFirstSettlement()
    {
        placeSetupSettlement(1);
    }

    @Override
    protected void placeSecondSettlement()
    {
        placeSetupSettlement(2);
    }

    /**
     * STAC's loop can enter an initial-placement state on a stale {@code ourTurn} flag (the state
     * message arrives before the turn message). Upstream it then sends a placement the server ignores;
     * we skip it without asking the policy, and log the skip.
     */
    private boolean notOurPlacementTurn(final String decision)
    {
        if (game.getCurrentPlayerNumber() == ourPlayerData.getPlayerNumber())
            return false;
        final JsonObject rec = new JsonObject();
        rec.addProperty("type", "out_of_turn_skip");
        rec.addProperty("game", game.getName());
        rec.addProperty("seat", ourPlayerData.getPlayerNumber());
        rec.addProperty("decision", decision);
        rec.addProperty("current", game.getCurrentPlayerNumber());
        rec.addProperty("state", game.getGameState());
        DecisionLog.write(rec);
        return true;
    }

    private void placeSetupSettlement(final int round)
    {
        if (notOurPlacementTurn("setup_settlement"))
            return;
        final JsonObject r = ask("setup_settlement", s -> s.addProperty("setup_round", round));
        if (cmd(r).equals("settlement"))
        {
            final int node = r.get("node").getAsInt();
            if (ourPlayerData.isPotentialSettlement(node))
            {
                lastStartingPieceCoord = node;
                client.putPiece(game, new SOCSettlement(ourPlayerData, node, null));
                return;
            }
            r.addProperty("reason", "setup settlement at node " + node + " not legal in STAC");
        }
        fallback("setup_settlement", r);
        if (round == 1)
            super.placeFirstSettlement();
        else
            super.placeSecondSettlement();
    }

    @Override
    public void placeInitRoad()
    {
        if (notOurPlacementTurn("setup_road"))
            return;
        final int round = (game.getGameState() == SOCGame.START1B) ? 1 : 2;
        final JsonObject r = ask("setup_road", s -> s.addProperty("setup_round", round));
        if (cmd(r).equals("road"))
        {
            final int edge = r.get("edge").getAsInt();
            final int settlement = ourPlayerData.getLastSettlementCoord();
            final int[] ends = soc.game.SOCBoard.getAdjacentNodesToEdge_arr(edge);
            if (ourPlayerData.isPotentialRoad(edge) && ((ends[0] == settlement) || (ends[1] == settlement)))
            {
                lastStartingPieceCoord = edge;
                lastStartingRoadTowardsNode = (ends[0] == settlement) ? ends[1] : ends[0];
                client.putPiece(game, new SOCRoad(ourPlayerData, edge, null));
                return;
            }
            r.addProperty("reason", "setup road at edge " + edge + " not legal in STAC");
        }
        fallback("setup_road", r);
        super.placeInitRoad();
    }

    // ------------------------------------------------------------------ robber, steal, discard

    @Override
    protected void moveRobber()
    {
        final JsonObject r = ask("robber", s -> s.addProperty("before_roll", game.getCurrentDice() == 0));
        if (cmd(r).equals("robber"))
        {
            final int hex = r.get("hex").getAsInt();
            if (game.canMoveRobber(ourPlayerData.getPlayerNumber(), hex))
            {
                client.moveRobber(game, ourPlayerData, hex);
                return;
            }
            r.addProperty("reason", "robber hex " + hex + " not legal in STAC");
        }
        fallback("robber", r);
        super.moveRobber();
    }

    @Override
    protected void chooseRobberVictim(final boolean[] choices)
    {
        final JsonObject r = ask("steal", s -> {
            s.addProperty("before_roll", game.getCurrentDice() == 0);
            final JsonArray v = new JsonArray();
            for (int pn = 0; pn < choices.length; ++pn)
                if (choices[pn])
                    v.add(pn);
            s.getAsJsonObject("js_legal").add("victims", v);
        });
        if (cmd(r).equals("steal"))
        {
            final int pn = r.get("victim").getAsInt();
            if ((pn >= 0) && (pn < choices.length) && choices[pn])
            {
                client.choosePlayer(game, pn);
                return;
            }
            r.addProperty("reason", "victim " + pn + " not offered by STAC");
        }
        fallback("steal", r);
        super.chooseRobberVictim(choices);
    }

    @Override
    protected void discard(final int numDiscards)
    {
        final JsonObject r = ask("discard", s -> s.addProperty("discard_count", numDiscards));
        if (cmd(r).equals("discard"))
        {
            final SOCResourceSet rs = StacStateJson.resourceSet(r.getAsJsonObject("resources"));
            if ((rs.getTotal() == numDiscards) && ourPlayerData.getResources().contains(rs))
            {
                client.discard(game, rs);
                return;
            }
            r.addProperty("reason", "discard set " + rs + " is not " + numDiscards + " held cards");
        }
        fallback("discard", r);
        super.discard(numDiscards);
    }

    // ------------------------------------------------------------------ turn flow

    @Override
    protected void handleGAMESTATE(final SOCGameState mes)
    {
        super.handleGAMESTATE(mes);
        final int st = game.getGameState();
        if ((st == SOCGame.PLAY1) || (st == SOCGame.PLAY))
        {
            // Road Building may end early (no legal edge or no road pieces left).
            expectPLACING_FREE_ROAD1 = false;
            expectPLACING_FREE_ROAD2 = false;
        }
        if ((st == SOCGame.OVER) && ! ended)
            recordEnd("gamestate_over");
    }

    @Override
    protected void rollOrPlayKnightOrExpectDice()
    {
        expectPLAY = false;
        if (waitingForOurTurn || ! ourTurn)
        {
            expectDICERESULT = true;
            return;
        }
        if (preRollCardPending)
        {
            // A card played before the roll has resolved; we are back at PLAY (roll or card).
            preRollCardPending = false;
            expectPLAY1 = false;
        }
        if (expectPLAY1 || expectDISCARD || expectPLACING_ROBBER || (expectDICERESULT && (counter < 4000)))
            return;
        final JsonObject r = ask("roll", null);
        final String c = cmd(r);
        if (c.equals("roll"))
        {
            expectDICERESULT = true;
            counter = 0;
            client.rollDice(game);
        } else if (c.equals("play_dev") && playCard(r, true)) {
            return;
        } else {
            fallback("roll", r);
            super.rollOrPlayKnightOrExpectDice();
        }
    }

    @Override
    protected void getActionForPLAY1()
    {
        if (game.getGameState() != SOCGame.PLAY1)
        {
            super.getActionForPLAY1();
            return;
        }
        if (game.getTurnCount() != turnSeen)
        {
            turnSeen = game.getTurnCount();
            decisionsThisTurn = 0;
            lastCommand = null;
            repeats = 0;
        }
        if (++decisionsThisTurn > MAX_DECISIONS_PER_TURN)
        {
            fallback("actions", refusal("more than " + MAX_DECISIONS_PER_TURN + " decisions this turn"));
            endOurTurn();
            return;
        }
        if (pendingOffer != null)
        {
            // The wait for an answer timed out (SOCRobotBrain cleared waitingForTradeResponse).
            chatTrade("offer_timeout", pendingTo, pendingOffer, null);
            clearPendingOffer();
        }
        final JsonObject r = ask("actions", null);
        final String c = cmd(r);
        final String key = r.has("action") ? r.get("action").toString() : c;
        if (key.equals(lastCommand) && ! c.equals("end_turn"))
            ++repeats;
        else
            repeats = 0;
        lastCommand = key;
        if (repeats >= MAX_REPEATS)
        {
            r.addProperty("reason", "command refused " + repeats + " times by STAC: " + key);
            fallback("actions", r);
            endOurTurn();
            return;
        }
        switch (c)
        {
        case "settlement":
            build(SOCPlayingPiece.SETTLEMENT, r.get("node").getAsInt(), r);
            break;
        case "city":
            build(SOCPlayingPiece.CITY, r.get("node").getAsInt(), r);
            break;
        case "road":
            build(SOCPlayingPiece.ROAD, r.get("edge").getAsInt(), r);
            break;
        case "buy_dev":
            waitingForDevCard = true;
            counter = 0;
            client.buyDevCard(game);
            break;
        case "bank":
            counter = 0;
            waitingForTradeMsg = true;
            client.bankTrade(game, StacStateJson.resourceSet(r.getAsJsonObject("give")),
                StacStateJson.resourceSet(r.getAsJsonObject("get")));
            break;
        case "play_dev":
            if (! playCard(r, false))
            {
                fallback("actions", r);
                super.getActionForPLAY1();
            }
            break;
        case "end_turn":
            endOurTurn();
            break;
        case "offer":
            if (StacRobotBrain.isChatNegotiation() && makeChatOffer(r))
                break;
            if (! r.has("reason"))
                r.addProperty("reason", StacRobotBrain.isChatNegotiation()
                    ? "offer not makeable in STAC" : "player trading is disabled for this match");
            fallback("actions", r);
            super.getActionForPLAY1();
            break;
        default:
            fallback("actions", r);
            super.getActionForPLAY1();
        }
    }

    // ------------------------------------------------------------------ chat trading

    private void clearPendingOffer()
    {
        pendingOffer = null;
        pendingTo = -1;
        ourPlayerData.setCurrentOffer(null);
    }

    /** Send the policy's offer in STAC's chat format; false (with a reason on r) if it cannot be made. */
    private boolean makeChatOffer(final JsonObject r)
    {
        final int pn = ourPlayerData.getPlayerNumber();
        final int to = r.get("to").getAsInt();
        final SOCResourceSet give = StacStateJson.resourceSet(r.getAsJsonObject("give"));
        final SOCResourceSet get = StacStateJson.resourceSet(r.getAsJsonObject("get"));
        if ((to < 0) || (to >= game.maxPlayers) || (to == pn) || (give.getTotal() == 0) || (get.getTotal() == 0)
            || ! ourPlayerData.getResources().contains(give))
        {
            r.addProperty("reason", "offer not makeable in STAC: recipient " + to + " give " + give + " get " + get);
            return false;
        }
        final boolean[] toArr = new boolean[game.maxPlayers];
        toArr[to] = true;
        final StacTradeOffer offer = new StacTradeOffer(game.getName(), pn, toArr, give, false, get, false);
        ourPlayerData.setCurrentOffer(offer);
        pendingOffer = offer;
        pendingTo = to;
        waitingForTradeResponse = true;
        counter = 0;
        client.sendText(game, new StacTradeMessage(Integer.toString(pn), Integer.toString(to), offer, false,
            new Persuasion(), "Would you trade?").toMessage());
        chatTrade("offer_sent", to, offer, null);
        return true;
    }

    private static boolean addressedTo(final StacTradeMessage tm, final int pn)
    {
        final String rec = tm.getReceivers();
        if (rec == null)
            return false;
        for (final String x : rec.split(","))
            if (x.trim().equals(Integer.toString(pn)))
                return true;
        return false;
    }

    private void reply(final boolean accept, final int to)
    {
        final int pn = ourPlayerData.getPlayerNumber();
        client.sendText(game, accept ? StacDialogueManager.toMessage(new StacTradeMessage(Integer.toString(pn),
                Integer.toString(to), true, false, false, false, null, false, "Deal.").toString())
            : StacDialogueManager.toMessage(new StacTradeMessage(Integer.toString(pn), Integer.toString(to),
                false, true, false, false, null, false, "No thanks.").toString()));
    }

    @Override
    protected void handleChat(final SOCGameTextMsg gtm)
    {
        final String text = gtm.getText();
        if (! StacRobotBrain.isChatNegotiation() || (text == null) || ! text.startsWith(StacTradeMessage.TRADE))
            return;
        final StacTradeMessage tm;
        final int sender;
        try
        {
            tm = StacTradeMessage.parse(StacDialogueManager.fromMessage(text));
            sender = tm.getSenderInt();
        } catch (RuntimeException e) {
            chatTrade("unparsed", -1, null, "unparsed chat trade message: " + e);
            return;
        }
        final int pn = ourPlayerData.getPlayerNumber();
        final int current = game.getCurrentPlayerNumber();
        if ((sender < 0) || (sender >= game.maxPlayers) || (sender == pn))
            return;
        if ((tm.isAccept() || tm.isReject()) && (sender != current))
        {
            synchronized (chatAnswers)
            {
                if (tm.isAccept())
                    ++chatAnswers[sender][0];
                ++chatAnswers[sender][1];
            }
        }
        if (! addressedTo(tm, pn))
            return;
        final StacTradeOffer offer = tm.getOffer();

        // An answer to our pending offer.
        if ((pendingOffer != null) && (sender == pendingTo))
        {
            final StacTradeOffer ours = pendingOffer;
            if (tm.isAccept())
            {
                reply(true, sender);  // the confirming accept; the server executes the exchange
                clearPendingOffer();
                waitingForTradeResponse = false;
                waitingForTradeMsg = true;
                counter = 0;
                chatTrade("offer_accepted", sender, ours, null);
                return;
            }
            if (tm.isReject() || tm.isNoResponse() || (offer != null))
            {
                if (offer != null)
                    reply(false, sender);  // we never take counter-offers
                clearPendingOffer();
                waitingForTradeResponse = false;
                counter = 0;
                chatTrade(offer != null ? "offer_countered" : (tm.isReject() ? "offer_rejected" : "offer_no_response"),
                    sender, ours, null);
                return;
            }
        }

        if (offer == null)
            return;  // accepts, rejects and no-responses to offers that are not ours
        // An offer to us: only a trade with the current player can execute.
        String refuse = null;
        if ((sender != current) && (pn != current))
            refuse = "offer not involving the current player";
        else if (offer.isPartialOffer() || offer.hasDisjunctiveGiveSet() || offer.hasDisjunctiveGetSet())
            refuse = "partial or disjunctive offer";
        else if (! ourPlayerData.getResources().contains(offer.getGetSet()))
            refuse = "we do not hold the requested resources";
        else if (pendingOffer != null)
            refuse = "our own offer is pending";
        if (refuse != null)
        {
            reply(false, sender);
            chatTrade("answered_reject", sender, offer, refuse);
            return;
        }
        final JsonObject r = ask("trade_response", s -> s.add("offer", StacStateJson.offer(offer)));
        final String c = cmd(r);
        if (! c.equals("accept") && ! c.equals("reject"))
        {
            r.addProperty("reason", "no accept/reject from the policy (" + c + "); rejected");
            fallback("trade_response", r);
        }
        final boolean accept = c.equals("accept");
        reply(accept, sender);
        chatTrade(accept ? "answered_accept" : "answered_reject", sender, offer, null);
    }

    /** One decision-log record per chat trade event. */
    private void chatTrade(final String event, final int other, final SOCTradeOffer offer, final String reason)
    {
        final JsonObject rec = new JsonObject();
        rec.addProperty("type", "chat_trade");
        rec.addProperty("game", game.getName());
        rec.addProperty("seat", ourPlayerData.getPlayerNumber());
        rec.addProperty("event", event);
        rec.addProperty("other", other);
        rec.addProperty("turn_count", game.getTurnCount());
        rec.addProperty("current", game.getCurrentPlayerNumber());
        if (offer != null)
            rec.add("offer", StacStateJson.offer(offer));
        if (reason != null)
            rec.addProperty("reason", reason);
        DecisionLog.write(rec);
    }

    /** End our turn exactly as {@code SOCRobotBrain.getActionForPLAY1} does. */
    private void endOurTurn()
    {
        if (pendingOffer != null)
            clearPendingOffer();
        waitingForGameState = true;
        counter = 0;
        expectPLAY = true;
        waitingForOurTurn = true;
        doneTrading = (robotParameters.getTradeFlag() != 1);
        negotiator.resetOffersMade();
        negotiator.resetTradesMade();
        resetBuildingPlan();
        negotiator.resetTargetPieces();
        client.endTurn(game);
    }

    private void build(final int pieceType, final int coord, final JsonObject r)
    {
        final SOCPlayingPiece piece;
        final boolean legal;
        switch (pieceType)
        {
        case SOCPlayingPiece.SETTLEMENT:
            piece = new SOCSettlement(ourPlayerData, coord, null);
            legal = ourPlayerData.isPotentialSettlement(coord);
            break;
        case SOCPlayingPiece.CITY:
            piece = new SOCCity(ourPlayerData, coord, null);
            legal = ourPlayerData.isPotentialCity(coord);
            break;
        default:
            piece = new SOCRoad(ourPlayerData, coord, null);
            legal = ourPlayerData.isPotentialRoad(coord);
        }
        if (! legal || piece.equals(whatWeFailedToBuild))
        {
            r.addProperty("reason", legal ? "STAC already refused this placement"
                : "placement at " + coord + " not legal in STAC");
            fallback("actions", r);
            super.getActionForPLAY1();
            return;
        }
        whatWeWantToBuild = piece;
        waitingForGameState = true;
        counter = 0;
        if (pieceType == SOCPlayingPiece.SETTLEMENT)
            expectPLACING_SETTLEMENT = true;
        else if (pieceType == SOCPlayingPiece.CITY)
            expectPLACING_CITY = true;
        else
            expectPLACING_ROAD = true;
        client.buildRequest(game, pieceType);
    }

    /** Play the development card named in a {@code play_dev} reply. False if STAC would refuse it. */
    private boolean playCard(final JsonObject r, final boolean beforeRoll)
    {
        final String card = r.get("card").getAsString();
        final int pn = ourPlayerData.getPlayerNumber();
        final int type;
        final boolean ok;
        switch (card)
        {
        case "knight": type = SOCDevCardConstants.KNIGHT; ok = game.canPlayKnight(pn); break;
        case "road_building": type = SOCDevCardConstants.ROADS; ok = game.canPlayRoadBuilding(pn); break;
        case "year_of_plenty": type = SOCDevCardConstants.DISC; ok = game.canPlayDiscovery(pn); break;
        case "monopoly": type = SOCDevCardConstants.MONO; ok = game.canPlayMonopoly(pn); break;
        default:
            r.addProperty("reason", "unknown card " + card);
            return false;
        }
        if (! ok)
        {
            r.addProperty("reason", "STAC does not allow playing " + card + " now");
            return false;
        }
        counter = 0;
        waitingForGameState = true;
        switch (type)
        {
        case SOCDevCardConstants.KNIGHT:
            expectPLACING_ROBBER = true;
            break;
        case SOCDevCardConstants.ROADS:
            expectPLACING_FREE_ROAD1 = true;
            freeRoadsBeforeRoll = beforeRoll;
            break;
        case SOCDevCardConstants.DISC:
            decisionMaker.resourceChoices = StacStateJson.resourceSet(r.getAsJsonObject("pick"));
            expectWAITING_FOR_DISCOVERY = true;
            break;
        default:
            decisionMaker.monopolyChoice = r.get("resource").getAsInt();
            expectWAITING_FOR_MONOPOLY = true;
        }
        if (beforeRoll)
            preRollCardPending = true;
        client.playDevCard(game, type);
        return true;
    }

    @Override
    protected void placeIfExpectPlacing()
    {
        if (waitingForGameState)
            return;
        final int gs = game.getGameState();
        if (((gs == SOCGame.PLACING_FREE_ROAD1) || (gs == SOCGame.PLACING_FREE_ROAD2))
            && ourTurn && ! waitingForOurTurn && (expectPLACING_FREE_ROAD1 || expectPLACING_FREE_ROAD2))
        {
            final boolean first = (gs == SOCGame.PLACING_FREE_ROAD1);
            final JsonObject r = ask("free_roads", s -> {
                s.addProperty("free_roads", first ? 2 : 1);
                s.addProperty("before_roll", freeRoadsBeforeRoll);
            });
            int edge = -1;
            if (cmd(r).equals("road"))
                edge = r.get("edge").getAsInt();
            if ((edge == -1) || ! ourPlayerData.isPotentialRoad(edge))
            {
                if (edge != -1)
                    r.addProperty("reason", "free road edge " + edge + " not legal in STAC");
                fallback("free_roads", r);
                edge = firstPotentialRoad();
            }
            counter = 0;
            waitingForGameState = true;
            if (first)
            {
                expectPLACING_FREE_ROAD1 = false;
                expectPLACING_FREE_ROAD2 = true;
            } else {
                expectPLACING_FREE_ROAD1 = false;
                expectPLACING_FREE_ROAD2 = false;
                expectPLAY1 = ! freeRoadsBeforeRoll;
            }
            client.putPiece(game, new SOCRoad(ourPlayerData, edge, null));
            return;
        }
        super.placeIfExpectPlacing();
    }

    /** STAC-legal fallback for a free road: the first potential road edge. */
    private int firstPotentialRoad()
    {
        final soc.game.SOCBoard board = game.getBoard();
        for (final int hex : board.getHexLandCoords())
        {
            final java.util.Vector<?> around = soc.game.SOCBoard.getAdjacentNodesToHex(hex);
            for (final Object oa : around)
                for (final Object ob : around)
                {
                    final int a = (Integer) oa, b = (Integer) ob;
                    if (a < b && StacStateJson.adjacent(a, b))
                    {
                        final int e = board.getEdgeBetweenAdjacentNodes(a, b);
                        if (ourPlayerData.isPotentialRoad(e))
                            return e;
                    }
                }
        }
        return -1;
    }

    @Override
    protected int considerOffer(final SOCTradeOffer offer)
    {
        // Protocol offers: only STAC's chat negotiation carries player trades in these matches.
        if (offer.getTo()[ourPlayerData.getPlayerNumber()])
            fallback("trade_response", refusal("protocol offer (not chat); left to STAC"));
        return super.considerOffer(offer);
    }

    // ------------------------------------------------------------------ game end

    private void recordEnd(final String how)
    {
        if (ended || ! started)
            return;
        ended = true;
        final JsonObject rec = new JsonObject();
        rec.addProperty("type", "game_end");
        rec.addProperty("game", gameName);
        rec.addProperty("seat", ourPN);
        rec.addProperty("how", how);
        final SOCGame g = game;
        if (g != null)
        {
            final SOCPlayer w = g.getPlayerWithWin();
            rec.addProperty("winner_declared", (w == null) ? -1 : w.getPlayerNumber());
            rec.addProperty("first_player", g.getFirstPlayer());
            rec.addProperty("turn_count", g.getTurnCount());
            rec.addProperty("rounds", g.getRoundCount());
        }
        DecisionLog.write(rec);
        try
        {
            final JsonObject end = new JsonObject();
            end.addProperty("kind", "end_game");
            end.addProperty("game", gameName);
            PythonBridge.get().call(end);
        } catch (Exception e) {
            System.err.println("catanai: end_game: " + e);
        }
    }

    @Override
    public void kill()
    {
        try
        {
            recordEnd("brain_killed");
        } catch (RuntimeException e) {
            System.err.println("catanai: recordEnd: " + e);
        }
        super.kill();
    }
}
