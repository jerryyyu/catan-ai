package catanai.jsettlers;

import java.io.FileWriter;
import java.io.IOException;
import java.io.Writer;

import com.google.gson.JsonObject;

/**
 * Append-only JSON-lines log of bridge decisions, fallbacks, errors and game results
 * (system property {@code catanai.decision.log}). The smoke script builds its receipt from it.
 */
public final class DecisionLog
{
    private static Writer out;
    private static boolean opened;

    public static synchronized void write(final JsonObject record)
    {
        if (! opened)
        {
            opened = true;
            final String path = System.getProperty("catanai.decision.log");
            if (path != null)
            {
                try
                {
                    out = new FileWriter(path, true);
                } catch (IOException e) {
                    System.err.println("catanai: cannot open decision log " + path + ": " + e);
                }
            }
        }
        record.addProperty("t", System.currentTimeMillis());
        if (out == null)
        {
            System.err.println("catanai: " + record);
            return;
        }
        try
        {
            out.write(record.toString());
            out.write('\n');
            out.flush();
        } catch (IOException e) {
            System.err.println("catanai: decision log write failed: " + e);
        }
    }
}
