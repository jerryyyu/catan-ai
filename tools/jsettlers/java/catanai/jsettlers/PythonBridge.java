package catanai.jsettlers;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import com.google.gson.JsonObject;
import com.google.gson.JsonParser;

/**
 * One long-lived Python bridge process ({@code scripts/jsettlers_bridge.py}) per JVM.
 * Requests and replies are single JSON lines; calls are serialized.
 *<P>
 * System properties: {@code catanai.bridge.cmd} (whitespace-separated command),
 * {@code catanai.bridge.cwd} (working directory), {@code catanai.bridge.stderr} (file for
 * the process's stderr; default inherit).
 */
public final class PythonBridge
{
    private static PythonBridge instance;

    private Process process;
    private BufferedWriter toPython;
    private BufferedReader fromPython;
    private long nextId = 1;

    public static synchronized PythonBridge get()
    {
        if (instance == null)
            instance = new PythonBridge();
        return instance;
    }

    private void start() throws IOException
    {
        final String cmd = System.getProperty("catanai.bridge.cmd");
        if (cmd == null)
            throw new IOException("system property catanai.bridge.cmd is not set");
        final List<String> args = new ArrayList<>(Arrays.asList(cmd.trim().split("\\s+")));
        final ProcessBuilder pb = new ProcessBuilder(args);
        final String cwd = System.getProperty("catanai.bridge.cwd");
        if (cwd != null)
            pb.directory(new File(cwd));
        final String err = System.getProperty("catanai.bridge.stderr");
        if (err != null)
            pb.redirectError(ProcessBuilder.Redirect.appendTo(new File(err)));
        else
            pb.redirectError(ProcessBuilder.Redirect.INHERIT);
        process = pb.start();
        toPython = new BufferedWriter(new OutputStreamWriter(process.getOutputStream(), StandardCharsets.UTF_8));
        fromPython = new BufferedReader(new InputStreamReader(process.getInputStream(), StandardCharsets.UTF_8));
    }

    /** Send one request and wait for its reply. Throws on any transport failure. */
    public synchronized JsonObject call(final JsonObject request) throws IOException
    {
        if ((process == null) || ! process.isAlive())
            start();
        final long id = nextId++;
        request.addProperty("id", id);
        toPython.write(request.toString());
        toPython.write('\n');
        toPython.flush();
        final String line = fromPython.readLine();
        if (line == null)
            throw new IOException("bridge process closed its output");
        final JsonObject reply = JsonParser.parseString(line).getAsJsonObject();
        if (! reply.has("id") || reply.get("id").isJsonNull() || reply.get("id").getAsLong() != id)
            throw new IOException("bridge reply id mismatch: " + line);
        return reply;
    }
}
