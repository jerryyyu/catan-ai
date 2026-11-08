package catanai.stac;

import java.util.LinkedHashSet;
import java.util.Set;
import java.util.Vector;

import com.google.gson.JsonArray;
import com.google.gson.JsonObject;

import soc.game.SOCBoard;
import soc.game.SOCCity;
import soc.game.SOCDevCardConstants;
import soc.game.SOCDevCardSet;
import soc.game.SOCGame;
import soc.game.SOCPlayer;
import soc.game.SOCPlayingPiece;
import soc.game.SOCResourceConstants;
import soc.game.SOCResourceSet;
import soc.game.SOCRoad;
import soc.game.SOCSettlement;
import soc.game.SOCTradeOffer;

/**
 * Serializes the STAC robot client's own copy of a game, as seen by one seat, in the snapshot
 * format of the Catan AI JSettlers bridge ({@code tools/jsettlers/jsbridge/state.py}), so the Python
 * side is shared.
 *<P>
 * Hidden-information boundary: the brain's {@link SOCGame} is the client-side copy the STAC server
 * keeps current for that seat. With FullyObservable off (the match runner refuses anything else, and
 * the patched server logs the flag per game) it holds, for opponents, only the resources the client
 * has seen them gain or spend plus an UNKNOWN count, and development cards as a count of UNKNOWN cards.
 * This serializer copies exactly that ({@code known}, {@code unknown}, {@code total}, {@code dev_count});
 * our own exact hand and card list go only in the separate {@code own} block.
 *<P>
 * STAC (JSettlers 1.1.16) codes are translated to the JSettlers2 codes the Python side expects:
 * desert hex 0 -> 6, development card KNIGHT 0 -> 9 (others unchanged).
 */
public final class StacStateJson
{
    private StacStateJson() {}

    public static final int[] RESOURCES = {
        SOCResourceConstants.CLAY, SOCResourceConstants.ORE, SOCResourceConstants.SHEEP,
        SOCResourceConstants.WHEAT, SOCResourceConstants.WOOD };

    /** JSettlers2 hex type of a STAC land hex type (desert is 0 in 1.1.x and 6 in 2.x). */
    static int hexType(final int stacType)
    {
        return (stacType == SOCBoard.DESERT_HEX) ? 6 : stacType;
    }

    /** JSettlers2 development card code of a STAC card type (KNIGHT is 0 in 1.1.x, 9 in 2.x). */
    static int devCode(final int stacType)
    {
        return (stacType == SOCDevCardConstants.KNIGHT) ? 9 : stacType;
    }

    public static JsonObject resources(final SOCResourceSet rs)
    {
        final JsonObject o = new JsonObject();
        for (final int r : RESOURCES)
        {
            final int n = rs.getAmount(r);
            if (n != 0)
                o.addProperty(Integer.toString(r), n);
        }
        return o;
    }

    public static SOCResourceSet resourceSet(final JsonObject o)
    {
        final SOCResourceSet rs = new SOCResourceSet();
        if (o != null)
            for (final java.util.Map.Entry<String, com.google.gson.JsonElement> e : o.entrySet())
                rs.add(e.getValue().getAsInt(), Integer.parseInt(e.getKey()));
        return rs;
    }

    private static Vector<?> v(final Vector<?> x)
    {
        return (x == null) ? new Vector<Object>() : x;
    }

    public static JsonObject board(final SOCBoard board)
    {
        final JsonObject out = new JsonObject();
        out.addProperty("encoding", board.getBoardEncodingFormat());
        final JsonArray hexes = new JsonArray(), edges = new JsonArray(), nodes = new JsonArray();
        final Set<Integer> seenNodes = new LinkedHashSet<Integer>(), seenEdges = new LinkedHashSet<Integer>();
        for (final int hex : board.getHexLandCoords())
        {
            final JsonObject h = new JsonObject();
            h.addProperty("coord", hex);
            h.addProperty("type", hexType(board.getHexTypeFromCoord(hex)));
            h.addProperty("number", board.getNumberOnHexFromCoord(hex));
            hexes.add(h);
            final Vector<?> around = v(SOCBoard.getAdjacentNodesToHex(hex));
            for (final Object o : around)
                seenNodes.add((Integer) o);
            for (final Object oa : around)
                for (final Object ob : around)
                {
                    final int a = (Integer) oa, b = (Integer) ob;
                    if (a >= b || ! adjacent(a, b))
                        continue;
                    final int e = board.getEdgeBetweenAdjacentNodes(a, b);
                    if (seenEdges.add(e))
                    {
                        final JsonObject edge = new JsonObject();
                        edge.addProperty("coord", e);
                        final int[] en = SOCBoard.getAdjacentNodesToEdge_arr(e);
                        final JsonArray ena = new JsonArray();
                        ena.add(en[0]);
                        ena.add(en[1]);
                        edge.add("nodes", ena);
                        edges.add(edge);
                    }
                }
        }
        for (final int n : seenNodes)
        {
            final JsonObject node = new JsonObject();
            node.addProperty("coord", n);
            final JsonArray around = new JsonArray();
            for (final Object h : v(SOCBoard.getAdjacentHexesToNode(n)))
                around.add((Integer) h);
            node.add("hexes", around);
            nodes.add(node);
        }
        final JsonArray ports = new JsonArray();
        // Classic 1.1.x boards: fixed port edges; the type comes from the node-to-port map the client
        // fills from the hex layout (portsLayout itself is not kept for the 4-player layout).
        final int[] portEdges = board.getPortsEdges();
        for (int i = 0; i < portEdges.length; ++i)
        {
            final JsonObject p = new JsonObject();
            p.addProperty("edge", portEdges[i]);
            final int[] en = SOCBoard.getAdjacentNodesToEdge_arr(portEdges[i]);
            p.addProperty("type", board.getPortTypeFromNodeCoord(en[0]));
            final JsonArray ena = new JsonArray();
            ena.add(en[0]);
            ena.add(en[1]);
            p.add("nodes", ena);
            p.addProperty("node_type", board.getPortTypeFromNodeCoord(en[0]));
            p.addProperty("node_type_b", board.getPortTypeFromNodeCoord(en[1]));
            ports.add(p);
        }
        out.add("hexes", hexes);
        out.add("nodes", nodes);
        out.add("edges", edges);
        out.add("ports", ports);
        out.addProperty("robber", board.getRobberHex());
        return out;
    }

    static boolean adjacent(final int a, final int b)
    {
        for (final int n : SOCBoard.getAdjacentNodesToNode_arr_v1(a))
            if (n == b)
                return true;
        return false;
    }

    public static JsonObject snapshot(final SOCGame game, final int ourPN, final String decision)
    {
        final JsonObject s = new JsonObject();
        s.addProperty("game", game.getName());
        s.addProperty("decision", decision);
        s.addProperty("seat", ourPN);
        s.addProperty("first_player", game.getFirstPlayer());
        s.addProperty("current", game.getCurrentPlayerNumber());
        s.addProperty("state", game.getGameState());
        s.addProperty("turn_count", game.getTurnCount());
        s.addProperty("round", game.getRoundCount());
        s.addProperty("dice", game.getCurrentDice());
        s.addProperty("dev_remaining", game.getNumDevCards());
        final SOCPlayer lr = game.getPlayerWithLongestRoad(), la = game.getPlayerWithLargestArmy();
        s.addProperty("longest_road", (lr == null) ? -1 : lr.getPlayerNumber());
        s.addProperty("largest_army", (la == null) ? -1 : la.getPlayerNumber());
        s.addProperty("vp_to_win", game.getVpWinner());
        s.add("board", board(game.getBoard()));

        final JsonArray players = new JsonArray();
        for (int pn = 0; pn < game.maxPlayers; ++pn)
        {
            final SOCPlayer pl = game.getPlayer(pn);
            final JsonObject p = new JsonObject();
            p.addProperty("pn", pn);
            final JsonArray settlements = new JsonArray(), cities = new JsonArray(), roads = new JsonArray();
            for (final Object st : v(pl.getSettlements()))
                settlements.add(((SOCSettlement) st).getCoordinates());
            for (final Object c : v(pl.getCities()))
                cities.add(((SOCCity) c).getCoordinates());
            for (final Object r : v(pl.getRoads()))
                roads.add(((SOCRoad) r).getCoordinates());
            p.add("settlements", settlements);
            p.add("cities", cities);
            p.add("roads", roads);
            final SOCResourceSet rs = pl.getResources();
            p.add("known", resources(rs));
            p.addProperty("unknown", rs.getAmount(SOCResourceConstants.UNKNOWN));
            p.addProperty("total", rs.getTotal());
            p.addProperty("dev_count", pl.getDevCards().getTotal());
            p.addProperty("knights", pl.getNumKnights());
            p.addProperty("public_vp", pl.getPublicVP());
            p.addProperty("played_dev", pl.hasPlayedDevCard());
            players.add(p);
        }
        s.add("players", players);

        final SOCPlayer us = game.getPlayer(ourPN);
        final JsonObject own = new JsonObject();
        own.add("resources", resources(us.getResources()));
        final JsonArray dev = new JsonArray();
        final SOCDevCardSet cards = us.getDevCards();
        for (int type = SOCDevCardConstants.MIN; type <= SOCDevCardConstants.MAX_KNOWN; ++type)
        {
            addCards(dev, type, cards.getAmount(SOCDevCardSet.NEW, type), "new");
            addCards(dev, type, cards.getAmount(SOCDevCardSet.OLD, type), "playable");
        }
        own.add("dev", dev);
        s.add("own", own);
        return s;
    }

    private static void addCards(final JsonArray out, final int type, final int n, final String state)
    {
        for (int i = 0; i < n; ++i)
        {
            final JsonObject o = new JsonObject();
            o.addProperty("type", devCode(type));
            o.addProperty("state", state);
            out.add(o);
        }
    }

    /** STAC's own legal sets for our seat, for the bridge's legality cross-check. */
    public static JsonObject legal(final SOCGame game, final int ourPN, final String decision)
    {
        final JsonObject legal = new JsonObject();
        final SOCPlayer us = game.getPlayer(ourPN);
        final SOCBoard board = game.getBoard();
        final Set<Integer> nodes = new LinkedHashSet<Integer>(), edges = new LinkedHashSet<Integer>();
        for (final int hex : board.getHexLandCoords())
        {
            final Vector<?> around = v(SOCBoard.getAdjacentNodesToHex(hex));
            for (final Object oa : around)
            {
                nodes.add((Integer) oa);
                for (final Object ob : around)
                {
                    final int a = (Integer) oa, b = (Integer) ob;
                    if (a < b && adjacent(a, b))
                        edges.add(board.getEdgeBetweenAdjacentNodes(a, b));
                }
            }
        }
        final JsonArray settlements = new JsonArray(), cities = new JsonArray(), roads = new JsonArray();
        for (final int node : nodes)
        {
            if (us.isPotentialSettlement(node))
                settlements.add(node);
            if (us.isPotentialCity(node))
                cities.add(node);
        }
        for (final int e : edges)
            if (us.isPotentialRoad(e))
                roads.add(e);
        if (us.getNumPieces(SOCPlayingPiece.SETTLEMENT) > 0)
            legal.add("settlements", settlements);
        if (us.getNumPieces(SOCPlayingPiece.CITY) > 0)
            legal.add("cities", cities);
        if (us.getNumPieces(SOCPlayingPiece.ROAD) > 0)
            legal.add("roads", roads);
        final JsonObject rates = new JsonObject();
        final boolean misc = us.getPortFlag(SOCBoard.MISC_PORT);
        for (final int r : RESOURCES)
            rates.addProperty(Integer.toString(r), us.getPortFlag(r) ? 2 : (misc ? 3 : 4));
        legal.add("rates", rates);
        if (decision.equals("robber"))
        {
            final JsonArray hexes = new JsonArray();
            for (final int hex : board.getHexLandCoords())
                if (game.canMoveRobber(ourPN, hex))
                    hexes.add(hex);
            legal.add("robber_hexes", hexes);
        }
        return legal;
    }

    public static JsonObject offer(final SOCTradeOffer offer)
    {
        final JsonObject o = new JsonObject();
        o.addProperty("from", offer.getFrom());
        final JsonArray to = new JsonArray();
        for (final boolean b : offer.getTo())
            to.add(b);
        o.add("to", to);
        o.add("give", resources(offer.getGiveSet()));
        o.add("get", resources(offer.getGetSet()));
        return o;
    }
}
