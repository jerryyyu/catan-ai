package catanai.jsettlers;

import com.google.gson.JsonArray;
import com.google.gson.JsonObject;

import soc.game.SOCBoard;

/**
 * Serializes a classic 4-player JSettlers board (v1 encoding) for the Python bridge.
 * Only public board facts: terrain, number tokens, ports and the robber hex.
 */
public final class BoardJson
{
    private BoardJson() {}

    public static JsonObject board(final SOCBoard board)
    {
        final JsonObject out = new JsonObject();
        out.addProperty("encoding", board.getBoardEncodingFormat());
        final JsonArray hexes = new JsonArray();
        final JsonArray hexNodes = new JsonArray();
        final JsonArray edges = new JsonArray();
        final java.util.Set<Integer> seenEdges = new java.util.HashSet<>();
        final java.util.Set<Integer> seenNodes = new java.util.LinkedHashSet<>();
        for (final int hex : board.getLandHexCoords())
        {
            final JsonObject h = new JsonObject();
            h.addProperty("coord", hex);
            h.addProperty("type", board.getHexTypeFromCoord(hex));
            h.addProperty("number", board.getNumberOnHexFromCoord(hex));
            hexes.add(h);
            final int[] nodes = board.getAdjacentNodesToHex_arr(hex);
            final JsonArray hn = new JsonArray();
            for (final int n : nodes)
            {
                hn.add(n);
                seenNodes.add(n);
            }
            final JsonObject entry = new JsonObject();
            entry.addProperty("hex", hex);
            entry.add("nodes", hn);
            hexNodes.add(entry);
            for (int i = 0; i < nodes.length; ++i)
            {
                final int a = nodes[i], b = nodes[(i + 1) % nodes.length];
                final int e = board.getEdgeBetweenAdjacentNodes(a, b);
                if (seenEdges.add(e))
                {
                    final JsonObject edge = new JsonObject();
                    edge.addProperty("coord", e);
                    final int[] en = board.getAdjacentNodesToEdge_arr(e);
                    final JsonArray ena = new JsonArray();
                    ena.add(en[0]);
                    ena.add(en[1]);
                    edge.add("nodes", ena);
                    edges.add(edge);
                }
            }
        }
        final JsonArray nodes = new JsonArray();
        for (final int n : seenNodes)
        {
            final JsonObject node = new JsonObject();
            node.addProperty("coord", n);
            final JsonArray around = new JsonArray();
            for (final int h : board.getAdjacentHexesToNode(n))
                around.add(h);
            node.add("hexes", around);
            nodes.add(node);
        }
        final JsonArray ports = new JsonArray();
        final int[] portEdges = board.getPortsEdges();
        final int[] portTypes = board.getPortsLayout();
        for (int i = 0; i < portEdges.length; ++i)
        {
            final JsonObject p = new JsonObject();
            p.addProperty("edge", portEdges[i]);
            final int[] en = board.getAdjacentNodesToEdge_arr(portEdges[i]);
            p.addProperty("type", (portTypes != null) ? portTypes[i] : board.getPortTypeFromNodeCoord(en[0]));
            final JsonArray ena = new JsonArray();
            ena.add(en[0]);
            ena.add(en[1]);
            p.add("nodes", ena);
            p.addProperty("node_type", board.getPortTypeFromNodeCoord(en[0]));
            p.addProperty("node_type_b", board.getPortTypeFromNodeCoord(en[1]));
            ports.add(p);
        }
        out.add("hexes", hexes);
        out.add("hex_nodes", hexNodes);
        out.add("nodes", nodes);
        out.add("edges", edges);
        out.add("ports", ports);
        out.addProperty("robber", board.getRobberHex());
        return out;
    }
}
