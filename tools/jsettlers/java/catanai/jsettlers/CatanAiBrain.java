package catanai.jsettlers;

import java.util.function.Consumer;

import com.google.gson.JsonArray;
import com.google.gson.JsonElement;
import com.google.gson.JsonObject;

import soc.game.SOCBoard;
import soc.game.SOCCity;
import soc.game.SOCDevCardConstants;
import soc.game.SOCGame;
import soc.game.SOCPlayer;
import soc.game.SOCPlayingPiece;
import soc.game.SOCResourceSet;
import soc.game.SOCRoad;
import soc.game.SOCSettlement;
import soc.game.SOCTradeOffer;
import soc.message.SOCMessage;
import soc.robot.DiscardStrategy;
import soc.robot.OpeningBuildStrategy;
import soc.robot.RobberStrategy;
import soc.robot.SOCBuildPlanStack;
import soc.robot.SOCRobotBrain;
import soc.robot.SOCRobotDM;
import soc.robot.SOCRobotNegotiator;
import soc.util.CappedQueue;
import soc.util.SOCRobotParameters;

/**
 * Robot brain whose every decision is asked of the Catan AI served policy through
 * {@link PythonBridge}. JSettlers keeps tracking game state and stays authoritative: the brain
 * only sends the chosen action through the normal client API.
 *<P>
 * Decisions bridged: initial settlements and roads (via the opening strategy), roll vs. playing a
 * development card before the roll, each PLAY1 action (build, buy, bank/port trade, play a card,
 * offer a trade, end turn), free roads from Road Building, robber placement and victim, discards,
 * and answers to trade offers made to us (accept or reject; never counter).
 *<P>
 * When the bridge cannot give a JSettlers-legal action, the inherited JSettlers logic decides and
 * the decision is logged as a fallback with its reason. Bridge transport/exception failures are
 * logged as errors (and also fall back so the game continues).
 */
public class CatanAiBrain extends SOCRobotBrain
{
    /** Hard cap on bridged PLAY1 decisions in one turn, against request loops. */
    static final int MAX_DECISIONS_PER_TURN = 80;
    /** Identical consecutive PLAY1 commands allowed before treating the command as refused. */
    static final int MAX_REPEATS = 4;

    private final CatanAiClient cli;
    private boolean preRollCardPending;
    private boolean freeRoadsBeforeRoll;
    private boolean gameEnded;
    private boolean started;
    private int decisionsThisTurn;
    private String lastCommand;
    private int repeats;

    public CatanAiBrain(final CatanAiClient rc, final SOCRobotParameters params, final SOCGame ga,
        final CappedQueue<SOCMessage> mq)
    {
        super(rc, params, ga, mq);
        cli = rc;
    }

    // ------------------------------------------------------------------ strategies

    @Override
    protected void setStrategyFields()
    {
        super.setStrategyFields();
        openingBuildStrategy = new BridgeOpening(game, ourPlayerData, this);
        robberStrategy = new BridgeRobber(game, ourPlayerData, this, rand);
        discardStrategy = new BridgeDiscard(game, ourPlayerData, this, rand);
    }

    @Override
    protected SOCRobotDM createDM()
    {
        return new BridgeDM(this);
    }

    /** Decision maker whose Year of Plenty picks can be set from the bridge. */
    static final class BridgeDM extends SOCRobotDM
    {
        BridgeDM(final SOCRobotBrain br) { super(br); }

        void setResourceChoices(final SOCResourceSet rs) { resourceChoices = rs; }
    }

    final class BridgeOpening extends OpeningBuildStrategy
    {
        BridgeOpening(final SOCGame ga, final SOCPlayer pl, final SOCRobotBrain br) { super(ga, pl, br); }

        @Override
        public int planInitialSettlements()
        {
            final JsonObject r = ask("setup_settlement", s -> s.addProperty("setup_round", 1));
            if (cmd(r).equals("settlement"))
                return r.get("node").getAsInt();
            fallback("setup_settlement", r);
            return super.planInitialSettlements();
        }

        @Override
        public int planSecondSettlement()
        {
            final JsonObject r = ask("setup_settlement", s -> s.addProperty("setup_round", 2));
            if (cmd(r).equals("settlement"))
                return r.get("node").getAsInt();
            fallback("setup_settlement", r);
            super.planInitialSettlements();
            return super.planSecondSettlement();
        }

        @Override
        public int planInitRoad()
        {
            final int round = (game.getGameState() == SOCGame.START1B) ? 1 : 2;
            final JsonObject r = ask("setup_road", s -> s.addProperty("setup_round", round));
            if (cmd(r).equals("road"))
                return r.get("edge").getAsInt();
            fallback("setup_road", r);
            return super.planInitRoad();
        }
    }

    final class BridgeRobber extends RobberStrategy
    {
        BridgeRobber(final SOCGame ga, final SOCPlayer pl, final SOCRobotBrain br, final java.util.Random rnd)
        {
            super(ga, pl, br, rnd);
        }

        @Override
        public int getBestRobberHex()
        {
            final JsonObject r = ask("robber", s -> s.addProperty("before_roll", game.getCurrentDice() == 0));
            if (cmd(r).equals("robber"))
                return r.get("hex").getAsInt();
            fallback("robber", r);
            return super.getBestRobberHex();
        }

        @Override
        public int chooseRobberVictim(final boolean[] isVictim, final boolean canChooseNone)
        {
            final JsonObject r = ask("steal", s -> {
                s.addProperty("before_roll", game.getCurrentDice() == 0);
                final JsonArray v = new JsonArray();
                for (int pn = 0; pn < isVictim.length; ++pn)
                    if (isVictim[pn])
                        v.add(pn);
                s.getAsJsonObject("js_legal").add("victims", v);
            });
            if (cmd(r).equals("steal"))
            {
                final int pn = r.get("victim").getAsInt();
                if ((pn >= 0) && (pn < isVictim.length) && isVictim[pn])
                    return pn;
                r.addProperty("reason", "victim " + pn + " not offered by JSettlers");
            }
            fallback("steal", r);
            return super.chooseRobberVictim(isVictim, canChooseNone);
        }
    }

    final class BridgeDiscard extends DiscardStrategy
    {
        BridgeDiscard(final SOCGame ga, final SOCPlayer pl, final SOCRobotBrain br, final java.util.Random rnd)
        {
            super(ga, pl, br, rnd);
        }

        @Override
        public SOCResourceSet discard(final int numDiscards, final SOCBuildPlanStack plan)
        {
            final JsonObject r = ask("discard", s -> s.addProperty("discard_count", numDiscards));
            if (cmd(r).equals("discard"))
            {
                final SOCResourceSet rs = StateJson.resourceSet(r.getAsJsonObject("resources"));
                if ((rs.getTotal() == numDiscards) && ourPlayerData.getResources().contains(rs))
                    return rs;
                r.addProperty("reason", "discard set " + rs + " is not " + numDiscards + " held cards");
            }
            fallback("discard", r);
            return super.discard(numDiscards, plan);
        }
    }

    // ------------------------------------------------------------------ bridge calls

    private static String cmd(final JsonObject r)
    {
        final JsonElement c = r.get("cmd");
        return (c == null || c.isJsonNull()) ? "error" : c.getAsString();
    }

    JsonObject ask(final String decision, final Consumer<JsonObject> extra)
    {
        if (! started)
        {
            started = true;
            cli.ourSeat.put(game.getName(), ourPlayerNumber);
            final JsonObject rec = new JsonObject();
            rec.addProperty("type", "game_start");
            rec.addProperty("game", game.getName());
            rec.addProperty("seat", ourPlayerNumber);
            rec.addProperty("first_player", game.getFirstPlayer());
            rec.addProperty("bot", client.getNickname());
            final JsonArray names = new JsonArray();
            for (int pn = 0; pn < game.maxPlayers; ++pn)
                names.add(game.getPlayer(pn).getName());
            rec.add("players", names);
            DecisionLog.write(rec);
        }
        final long t0 = System.nanoTime();
        JsonObject reply;
        try
        {
            final JsonObject snap = StateJson.snapshot(game, ourPlayerNumber, decision);
            snap.addProperty("kind", "decide");
            snap.add("js_legal", StateJson.legal(game, ourPlayerNumber, decision));
            final int[][] a = cli.answers(game.getName());
            final JsonArray answers = new JsonArray();
            synchronized (a)
            {
                for (final int[] row : a)
                {
                    final JsonArray x = new JsonArray();
                    x.add(row[0]);
                    x.add(row[1]);
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
        rec.addProperty("seat", ourPlayerNumber);
        rec.addProperty("decision", decision);
        rec.addProperty("turn_count", game.getTurnCount());
        rec.addProperty("cmd", cmd(reply));
        rec.addProperty("latency_ms", Math.round(ms * 100) / 100.0);
        for (final String k : new String[] { "think_ms", "reason", "error", "legal_mismatches", "action" })
            if (reply.has(k))
                rec.add(k, reply.get(k));
        DecisionLog.write(rec);
        return reply;
    }

    /** Record that JSettlers' own logic made this decision instead of the policy. */
    void fallback(final String decision, final JsonObject reply)
    {
        final JsonObject rec = new JsonObject();
        rec.addProperty("type", "fallback");
        rec.addProperty("game", game.getName());
        rec.addProperty("seat", ourPlayerNumber);
        rec.addProperty("decision", decision);
        rec.addProperty("turn_count", game.getTurnCount());
        rec.addProperty("cmd", cmd(reply));
        rec.addProperty("reason", reply.has("reason") ? reply.get("reason").getAsString()
            : (reply.has("error") ? "bridge error: " + reply.get("error").getAsString() : "unhandled cmd " + cmd(reply)));
        if (reply.has("action"))
            rec.add("action", reply.get("action"));
        DecisionLog.write(rec);
    }

    // ------------------------------------------------------------------ turn flow

    @Override
    protected void resetFieldsAndBuildingPlan()
    {
        super.resetFieldsAndBuildingPlan();
        decisionsThisTurn = 0;
        lastCommand = null;
        repeats = 0;
    }

    @Override
    protected void handleGAMESTATE(final int newState)
    {
        super.handleGAMESTATE(newState);
        if ((newState == SOCGame.PLAY1) || (newState == SOCGame.ROLL_OR_CARD))
        {
            // Road Building may end early (no legal edge or no road pieces left).
            expectPLACING_FREE_ROAD1 = false;
            expectPLACING_FREE_ROAD2 = false;
        }
        if ((newState == SOCGame.OVER) && ! gameEnded)
        {
            gameEnded = true;
            final SOCPlayer w = game.getPlayerWithWin();
            cli.recordGameEnd(game.getName(), game.getCurrentPlayerNumber(), (w == null) ? -1 : w.getPlayerNumber());
        }
    }

    @Override
    protected void rollOrPlayKnightOrExpectDice()
    {
        expectROLL_OR_CARD = false;
        if (waitingForOurTurn || ! ourTurn)
        {
            expectDICERESULT = true;
            return;
        }
        if (preRollCardPending)
        {
            // A development card played before the roll has resolved; we are back at ROLL_OR_CARD.
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
    protected void planAndDoActionForPLAY1()
    {
        if (game.getGameState() != SOCGame.PLAY1)
        {
            super.planAndDoActionForPLAY1();
            return;
        }
        if (++decisionsThisTurn > MAX_DECISIONS_PER_TURN)
        {
            final JsonObject r = new JsonObject();
            r.addProperty("cmd", "fallback");
            r.addProperty("reason", "more than " + MAX_DECISIONS_PER_TURN + " decisions this turn");
            fallback("actions", r);
            endOurTurn();
            return;
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
            r.addProperty("reason", "command refused " + repeats + " times by JSettlers: " + key);
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
            client.buyDevCard(game);
            break;
        case "bank":
            counter = 0;
            waitingForTradeMsg = true;
            client.bankTrade(game, StateJson.resourceSet(r.getAsJsonObject("give")),
                StateJson.resourceSet(r.getAsJsonObject("get")));
            break;
        case "play_dev":
            if (! playCard(r, false))
            {
                fallback("actions", r);
                super.planAndDoActionForPLAY1();
            }
            break;
        case "offer":
            makeBridgeOffer(r);
            break;
        case "end_turn":
            endOurTurn();
            break;
        default:
            fallback("actions", r);
            super.planAndDoActionForPLAY1();
        }
    }

    private void endOurTurn()
    {
        resetFieldsAtEndTurn();
        client.endTurn(game);
    }

    private void build(final int pieceType, final int coord, final JsonObject r)
    {
        final SOCPlayingPiece piece;
        switch (pieceType)
        {
        case SOCPlayingPiece.SETTLEMENT:
            piece = new SOCSettlement(ourPlayerData, coord, null);
            break;
        case SOCPlayingPiece.CITY:
            piece = new SOCCity(ourPlayerData, coord, null);
            break;
        default:
            piece = new SOCRoad(ourPlayerData, coord, null);
        }
        if (piece.equals(whatWeFailedToBuild))
        {
            r.addProperty("reason", "JSettlers already refused this placement");
            fallback("actions", r);
            endOurTurn();
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

    /** Play the development card named in a {@code play_dev} reply. False if it cannot be sent. */
    private boolean playCard(final JsonObject r, final boolean beforeRoll)
    {
        final String card = r.get("card").getAsString();
        final int type;
        switch (card)
        {
        case "knight": type = SOCDevCardConstants.KNIGHT; break;
        case "road_building": type = SOCDevCardConstants.ROADS; break;
        case "year_of_plenty": type = SOCDevCardConstants.DISC; break;
        case "monopoly": type = SOCDevCardConstants.MONO; break;
        default:
            r.addProperty("reason", "unknown card " + card);
            return false;
        }
        if (type == rejectedPlayDevCardType)
        {
            r.addProperty("reason", "JSettlers refused " + card + " earlier this turn");
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
            ((BridgeDM) decisionMaker).setResourceChoices(StateJson.resourceSet(r.getAsJsonObject("pick")));
            expectWAITING_FOR_DISCOVERY = true;
            break;
        default:
            monopolyStrategy.setMonopolyChoice(r.get("resource").getAsInt());
            expectWAITING_FOR_MONOPOLY = true;
        }
        if (beforeRoll)
            preRollCardPending = true;
        client.playDevCard(game, type);
        return true;
    }

    private void makeBridgeOffer(final JsonObject r)
    {
        final int to = r.get("to").getAsInt();
        final boolean[] toArr = new boolean[game.maxPlayers];
        toArr[to] = true;
        final SOCTradeOffer offer = new SOCTradeOffer(game.getName(), ourPlayerNumber, toArr,
            StateJson.resourceSet(r.getAsJsonObject("give")), StateJson.resourceSet(r.getAsJsonObject("get")));
        ourPlayerData.setCurrentOffer(offer);
        for (int pn = 0; pn < game.maxPlayers; ++pn)
            offerRejections[pn] = false;
        waitingForTradeResponse = true;
        tradeResponseTimeoutSec = TRADE_RESPONSE_TIMEOUT_SEC_BOTS_ONLY;
        counter = 0;
        client.offerTrade(game, offer);
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
                    r.addProperty("reason", "free road edge " + edge + " not legal in JSettlers");
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

    /** JSettlers-legal fallback for a free road: the first potential road edge. */
    private int firstPotentialRoad()
    {
        final SOCBoard board = game.getBoard();
        for (final int hex : board.getLandHexCoords())
        {
            final int[] nodes = board.getAdjacentNodesToHex_arr(hex);
            for (int i = 0; i < nodes.length; ++i)
            {
                final int e = board.getEdgeBetweenAdjacentNodes(nodes[i], nodes[(i + 1) % nodes.length]);
                if (ourPlayerData.isPotentialRoad(e))
                    return e;
            }
        }
        return -1;
    }

    @Override
    protected int considerOffer(final SOCTradeOffer offer)
    {
        if (! offer.getTo()[ourPlayerNumber])
            return SOCRobotNegotiator.IGNORE_OFFER;
        final JsonObject r = ask("trade_response", s -> s.add("offer", StateJson.offer(offer)));
        final String c = cmd(r);
        if (c.equals("accept"))
            return SOCRobotNegotiator.ACCEPT_OFFER;
        if (c.equals("reject"))
            return SOCRobotNegotiator.REJECT_OFFER;
        fallback("trade_response", r);
        return super.considerOffer(offer);
    }
}
