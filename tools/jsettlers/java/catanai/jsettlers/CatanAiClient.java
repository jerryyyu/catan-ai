package catanai.jsettlers;

import java.util.concurrent.ConcurrentHashMap;

import soc.baseclient.ServerConnectInfo;
import soc.game.SOCGame;
import com.google.gson.JsonArray;
import com.google.gson.JsonObject;

import soc.game.SOCPlayer;
import soc.message.SOCAcceptOffer;
import soc.message.SOCGameElements;
import soc.message.SOCGameState;
import soc.message.SOCMessage;
import soc.message.SOCRejectOffer;
import soc.message.SOCTurn;
import soc.robot.SOCRobotBrain;
import soc.robot.SOCRobotClient;
import soc.util.CappedQueue;
import soc.util.SOCFeatureSet;
import soc.util.SOCRobotParameters;

/**
 * Third-party JSettlers robot client whose decisions come from the Catan AI served policy
 * through {@link PythonBridge}. Classic 4-player games only (no 6-player or sea features).
 * Can be auto-started by the server: {@code jsettlers.bots.start3p=1,catanai.jsettlers.CatanAiClient}.
 */
public class CatanAiClient extends SOCRobotClient
{
    /** Per game: for each player number, {accepted, answered} public answers to trade offers. */
    final ConcurrentHashMap<String, int[][]> tradeAnswers = new ConcurrentHashMap<>();

    public CatanAiClient(final ServerConnectInfo sci, final String nn, final String pw)
        throws IllegalArgumentException
    {
        super(sci, nn, pw);
        rbclass = CatanAiClient.class.getName();
    }

    @Override
    protected SOCFeatureSet buildClientFeats()
    {
        return new SOCFeatureSet(false, false);
    }

    @Override
    public SOCRobotBrain createBrain
        (final SOCRobotParameters params, final SOCGame ga, final CappedQueue<SOCMessage> mq)
    {
        tradeAnswers.put(ga.getName(), new int[ga.maxPlayers][2]);
        return new CatanAiBrain(this, params, ga, mq);
    }

    /** Per game: last current player number announced by the server. */
    final ConcurrentHashMap<String, Integer> currentPlayer = new ConcurrentHashMap<>();
    /** Per game: our seat, once the brain knows it. */
    final ConcurrentHashMap<String, Integer> ourSeat = new ConcurrentHashMap<>();
    private final java.util.Set<String> ended = ConcurrentHashMap.newKeySet();

    /**
     * Log the end of a game once. Bot-only games are destroyed right after GAMESTATE(OVER), so the
     * brain thread may never process that message; the client thread sees it first.
     * The winner is the current player when the game ends (only the current player can win).
     */
    void recordGameEnd(final String gaName, final int winner, final int winnerDeclared)
    {
        if (! ended.add(gaName))
            return;
        final SOCGame ga = games.get(gaName);
        final JsonObject rec = new JsonObject();
        rec.addProperty("type", "game_end");
        rec.addProperty("game", gaName);
        rec.addProperty("seat", ourSeat.getOrDefault(gaName, -1));
        rec.addProperty("winner", winner);
        rec.addProperty("winner_declared", winnerDeclared);
        if (ga != null)
        {
            rec.addProperty("first_player", ga.getFirstPlayer());
            rec.addProperty("turn_count", ga.getTurnCount());
            rec.addProperty("rounds", ga.getRoundCount());
            final JsonArray vp = new JsonArray(), pub = new JsonArray();
            for (int pn = 0; pn < ga.maxPlayers; ++pn)
            {
                vp.add(ga.getPlayer(pn).getTotalVP());
                pub.add(ga.getPlayer(pn).getPublicVP());
            }
            rec.add("total_vp_seen", vp);
            rec.add("public_vp", pub);
        }
        DecisionLog.write(rec);
        try
        {
            final JsonObject end = new JsonObject();
            end.addProperty("kind", "end_game");
            end.addProperty("game", gaName);
            PythonBridge.get().call(end);
        } catch (Exception e) {
            System.err.println("catanai: end_game: " + e);
        }
    }

    int[][] answers(final String game)
    {
        return tradeAnswers.computeIfAbsent(game, k -> new int[4][2]);
    }

    @Override
    public void treat(final SOCMessage mes)
    {
        if (mes instanceof SOCAcceptOffer)
        {
            final SOCAcceptOffer m = (SOCAcceptOffer) mes;
            final int pn = m.getAcceptingNumber();
            if ((pn >= 0) && (pn < 4))
            {
                final int[][] a = answers(m.getGame());
                synchronized (a) { a[pn][0]++; a[pn][1]++; }
            }
        } else if (mes instanceof SOCRejectOffer) {
            final SOCRejectOffer m = (SOCRejectOffer) mes;
            final int pn = m.getPlayerNumber();
            if ((pn >= 0) && (pn < 4) && (m.getReasonCode() == 0))
            {
                final int[][] a = answers(m.getGame());
                synchronized (a) { a[pn][1]++; }
            }
        } else if (mes instanceof SOCTurn) {
            currentPlayer.put(((SOCTurn) mes).getGame(), ((SOCTurn) mes).getPlayerNumber());
        } else if (mes instanceof SOCGameElements) {
            final SOCGameElements m = (SOCGameElements) mes;
            final int[] types = m.getElementTypes(), values = m.getValues();
            for (int i = 0; i < types.length; ++i)
                if (types[i] == SOCGameElements.GEType.CURRENT_PLAYER.getValue())
                    currentPlayer.put(m.getGame(), values[i]);
        } else if ((mes instanceof SOCGameState) && (((SOCGameState) mes).getState() == SOCGame.OVER)) {
            final String gaName = ((SOCGameState) mes).getGame();
            final SOCGame ga = games.get(gaName);
            final SOCPlayer w = (ga != null) ? ga.getPlayerWithWin() : null;
            recordGameEnd(gaName, currentPlayer.getOrDefault(gaName, -1), (w == null) ? -1 : w.getPlayerNumber());
        }
        super.treat(mes);
    }

    public static void main(final String[] args)
    {
        if (args.length < 5)
        {
            System.err.println("usage: java " + CatanAiClient.class.getName()
                + " hostname port_number bot_nickname password cookie");
            return;
        }
        final CatanAiClient cli = new CatanAiClient
            (new ServerConnectInfo(args[0], Integer.parseInt(args[1]), args[4]), args[2], args[3]);
        cli.init();
    }
}
