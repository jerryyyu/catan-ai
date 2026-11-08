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
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.TimeoutException;

import com.google.gson.JsonObject;
import com.google.gson.JsonParser;

/**
 * One long-lived Python bridge process ({@code scripts/jsettlers_bridge.py}) per JVM.
 * Requests and replies are single JSON lines; calls are serialized.
 *<P>
 * Every request (writing it, flushing it and reading the reply) has a deadline ({@code catanai.bridge.timeout_ms}, default 30000). A request that
 * gets no reply in time, or any transport failure, <b>retires</b> the transport: the process and
 * its descendants are destroyed and every later call fails at once with an {@link IOException}
 * (the brain counts each as a bridge error and falls back). The bridge is never silently restarted,
 * because a fresh process would lose the per-game policy state.
 *<P>
 * System properties: {@code catanai.bridge.cmd} (whitespace-separated command),
 * {@code catanai.bridge.cwd} (working directory), {@code catanai.bridge.stderr} (file for
 * the process's stderr; default inherit), {@code catanai.bridge.timeout_ms}.
 */
public final class PythonBridge
{
    private static PythonBridge instance;

    private Process process;
    private BufferedWriter toPython;
    private BufferedReader fromPython;
    private ExecutorService io;
    private long nextId = 1;
    private String retiredReason;
    private final long timeoutMs;

    PythonBridge(final long timeoutMs)
    {
        this.timeoutMs = timeoutMs;
    }

    public static synchronized PythonBridge get()
    {
        if (instance == null)
            instance = new PythonBridge(Long.getLong("catanai.bridge.timeout_ms", 30000L));
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
        io = Executors.newSingleThreadExecutor(r -> {
            final Thread t = new Thread(r, "catanai-bridge-io");
            t.setDaemon(true);
            return t;
        });
    }

    /** True once a stalled or broken transport has been retired. */
    public synchronized boolean isRetired()
    {
        return retiredReason != null;
    }

    /** The bridge process, for supervision and tests; null before the first call. */
    synchronized Process process()
    {
        return process;
    }

    private void retire(final String reason)
    {
        retiredReason = reason;
        if (process != null)
        {
            process.descendants().forEach(ProcessHandle::destroyForcibly);
            process.destroyForcibly();
        }
        if (io != null)
            io.shutdownNow();
        final JsonObject rec = new JsonObject();
        rec.addProperty("type", "bridge_retired");
        rec.addProperty("reason", reason);
        DecisionLog.write(rec);
    }

    /** Send one request and wait (bounded) for its reply. Throws on any transport failure. */
    public synchronized JsonObject call(final JsonObject request) throws IOException
    {
        if (retiredReason != null)
            throw new IOException("bridge retired: " + retiredReason);
        final long id = nextId++;
        final String line;
        try
        {
            if (process == null)
                start();
            if (! process.isAlive())
                throw new IOException("bridge process exited with " + process.exitValue());
            request.addProperty("id", id);
            final String payload = request.toString();
            // The whole exchange runs on the I/O thread under one deadline: a child that stops
            // reading can fill the stdin pipe and block write/flush just as a silent child blocks
            // readLine. On timeout retire() destroys the process, which unblocks the I/O thread.
            final Future<String> reply = io.submit(() -> {
                toPython.write(payload);
                toPython.write('\n');
                toPython.flush();
                return fromPython.readLine();
            });
            try
            {
                line = reply.get(timeoutMs, TimeUnit.MILLISECONDS);
            } catch (TimeoutException e) {
                reply.cancel(true);
                throw new IOException("no bridge reply within " + timeoutMs + " ms (request write + reply read)");
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                throw new IOException("interrupted waiting for the bridge");
            } catch (java.util.concurrent.ExecutionException e) {
                throw new IOException("bridge exchange failed: " + e.getCause());
            }
            if (line == null)
                throw new IOException("bridge process closed its output");
        } catch (IOException e) {
            retire(e.getMessage());
            throw e;
        }
        final JsonObject parsed;
        try
        {
            parsed = JsonParser.parseString(line).getAsJsonObject();
        } catch (RuntimeException e) {
            retire("unparseable bridge reply: " + e);
            throw new IOException("unparseable bridge reply: " + e);
        }
        if (! parsed.has("id") || parsed.get("id").isJsonNull() || parsed.get("id").getAsLong() != id)
        {
            retire("bridge reply id mismatch: " + line);
            throw new IOException("bridge reply id mismatch: " + line);
        }
        return parsed;
    }
}
