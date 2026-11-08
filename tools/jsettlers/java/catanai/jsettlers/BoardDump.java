package catanai.jsettlers;

import com.google.gson.GsonBuilder;
import com.google.gson.JsonArray;

import soc.game.SOCGame;

/**
 * Generates classic 4-player boards with JSettlers' own board generator and prints them
 * as a JSON array. Usage: {@code java catanai.jsettlers.BoardDump COUNT}.
 * Used to build the board-mapping round-trip fixture.
 */
public final class BoardDump
{
    public static void main(final String[] args)
    {
        final int count = (args.length > 0) ? Integer.parseInt(args[0]) : 50;
        final JsonArray boards = new JsonArray();
        for (int i = 0; i < count; ++i)
        {
            final SOCGame game = new SOCGame("boarddump-" + i);
            game.getBoard().makeNewBoard(null);
            boards.add(BoardJson.board(game.getBoard()));
        }
        System.out.println(new GsonBuilder().create().toJson(boards));
    }
}
