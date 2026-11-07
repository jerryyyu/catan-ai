package catanai.jsettlers;

import java.util.List;

import com.google.gson.JsonArray;
import com.google.gson.JsonObject;

import soc.game.SOCBoard;
import soc.game.SOCCity;
import soc.game.SOCGame;
import soc.game.SOCInventory;
import soc.game.SOCInventoryItem;
import soc.game.SOCPlayer;
import soc.game.SOCPlayingPiece;
import soc.game.SOCResourceConstants;
import soc.game.SOCResourceSet;
import soc.game.SOCRoutePiece;
import soc.game.SOCSettlement;
import soc.game.SOCTradeOffer;

/**
 * Serializes the robot client's own copy of a game, as seen by one seat, for the Python bridge.
 *<P>
 * Hidden-information boundary: the robot's {@link SOCGame} is the client-side copy the server
 * sends to that player. For opponents it holds only the resources the client has seen them
 * gain or spend (known amounts) plus an UNKNOWN count, and unplayed development cards as a
 * count. This serializer copies exactly that: {@code known}, {@code unknown}, {@code total},
 * {@code dev_count}. Our own exact hand and card list go only in the separate {@code own} block.
 */
public final class StateJson
{
    private StateJson() {}

    public static final int[] RESOURCES = {
        SOCResourceConstants.CLAY, SOCResourceConstants.ORE, SOCResourceConstants.SHEEP,
        SOCResourceConstants.WHEAT, SOCResourceConstants.WOOD };

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
            for (final String k : o.keySet())
                rs.add(o.get(k).getAsInt(), Integer.parseInt(k));
        return rs;
    }

    public static JsonObject snapshot(final SOCGame game, final int ourPN, final String decision)
    {
        final JsonObject s = new JsonObject();
        final SOCBoard board = game.getBoard();
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
        s.addProperty("vp_to_win", game.vp_winner);
        s.add("board", BoardJson.board(board));

        final JsonArray players = new JsonArray();
        for (int pn = 0; pn < game.maxPlayers; ++pn)
        {
            final SOCPlayer pl = game.getPlayer(pn);
            final JsonObject p = new JsonObject();
            p.addProperty("pn", pn);
            final JsonArray settlements = new JsonArray(), cities = new JsonArray(), roads = new JsonArray();
            for (final SOCSettlement st : pl.getSettlements())
                settlements.add(st.getCoordinates());
            for (final SOCCity c : pl.getCities())
                cities.add(c.getCoordinates());
            for (final SOCRoutePiece r : pl.getRoadsAndShips())
                if (r.getType() == SOCPlayingPiece.ROAD)
                    roads.add(r.getCoordinates());
            p.add("settlements", settlements);
            p.add("cities", cities);
            p.add("roads", roads);
            final SOCResourceSet rs = pl.getResources();
            p.add("known", resources(rs));
            p.addProperty("unknown", rs.getAmount(SOCResourceConstants.UNKNOWN));
            p.addProperty("total", rs.getTotal());
            p.addProperty("dev_count", pl.getInventory().getTotal());
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
        final SOCInventory inv = us.getInventory();
        addItems(dev, inv.getByState(SOCInventory.NEW), "new");
        addItems(dev, inv.getByState(SOCInventory.PLAYABLE), "playable");
        addItems(dev, inv.getByState(SOCInventory.KEPT), "kept");
        own.add("dev", dev);
        s.add("own", own);
        return s;
    }

    private static void addItems(final JsonArray out, final List<SOCInventoryItem> items, final String state)
    {
        if (items == null)
            return;
        for (final SOCInventoryItem it : items)
        {
            final JsonObject o = new JsonObject();
            o.addProperty("type", it.itype);
            o.addProperty("state", state);
            out.add(o);
        }
    }

    /** JSettlers' own legal sets for our seat, for the bridge's legality cross-check. */
    public static JsonObject legal(final SOCGame game, final int ourPN, final String decision)
    {
        final JsonObject legal = new JsonObject();
        final SOCPlayer us = game.getPlayer(ourPN);
        final SOCBoard board = game.getBoard();
        final JsonArray settlements = new JsonArray(), cities = new JsonArray(), roads = new JsonArray();
        for (final int hex : board.getLandHexCoords())
            for (final int node : board.getAdjacentNodesToHex_arr(hex))
            {
                if (contains(settlements, node) || contains(cities, node))
                    continue;
                if (us.canPlaceSettlement(node))
                    settlements.add(node);
                if (us.isPotentialCity(node))
                    cities.add(node);
            }
        for (final int hex : board.getLandHexCoords())
        {
            final int[] nodes = board.getAdjacentNodesToHex_arr(hex);
            for (int i = 0; i < nodes.length; ++i)
            {
                final int e = board.getEdgeBetweenAdjacentNodes(nodes[i], nodes[(i + 1) % nodes.length]);
                if (! contains(roads, e) && us.isPotentialRoad(e))
                    roads.add(e);
            }
        }
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
            for (final int hex : board.getLandHexCoords())
                if (game.canMoveRobber(ourPN, hex))
                    hexes.add(hex);
            legal.add("robber_hexes", hexes);
        }
        return legal;
    }

    private static boolean contains(final JsonArray a, final int v)
    {
        for (int i = 0; i < a.size(); ++i)
            if (a.get(i).getAsInt() == v)
                return true;
        return false;
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
