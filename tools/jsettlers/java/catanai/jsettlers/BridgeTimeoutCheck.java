package catanai.jsettlers;

import java.io.IOException;

import com.google.gson.JsonObject;

/**
 * Self-check for {@link PythonBridge}'s request deadline, run by the test suite against a fake
 * bridge command that reads requests but never replies. Prints one JSON line:
 * the first call's elapsed time and error, whether the process tree is gone, and the second
 * call's error (which must be immediate, without touching the retired transport).
 * Usage: {@code java -Dcatanai.bridge.cmd=... catanai.jsettlers.BridgeTimeoutCheck TIMEOUT_MS [PAD_BYTES]};
 * {@code PAD_BYTES} adds an oversized field to the first request (to fill a pipe the child never reads).
 */
public final class BridgeTimeoutCheck
{
    public static void main(final String[] args) throws Exception
    {
        final PythonBridge bridge = new PythonBridge(Long.parseLong(args[0]));
        final JsonObject out = new JsonObject();
        final long t0 = System.nanoTime();
        try
        {
            final JsonObject ping = new JsonObject();
            ping.addProperty("kind", "ping");
            if (args.length > 1)
            {
                final char[] pad = new char[Integer.parseInt(args[1])];
                java.util.Arrays.fill(pad, 'x');
                ping.addProperty("pad", new String(pad));
            }
            bridge.call(ping);
            out.addProperty("first_error", (String) null);
        } catch (IOException e) {
            out.addProperty("first_error", e.getMessage());
        }
        out.addProperty("first_ms", (System.nanoTime() - t0) / 1e6);
        final Process p = bridge.process();
        if (p != null)
            p.waitFor(5, java.util.concurrent.TimeUnit.SECONDS);
        out.addProperty("process_alive", (p != null) && p.isAlive());
        out.addProperty("descendants_alive", (p != null) ? p.descendants().filter(ProcessHandle::isAlive).count() : 0);
        out.addProperty("retired", bridge.isRetired());
        final long t1 = System.nanoTime();
        try
        {
            bridge.call(new JsonObject());
            out.addProperty("second_error", (String) null);
        } catch (IOException e) {
            out.addProperty("second_error", e.getMessage());
        }
        out.addProperty("second_ms", (System.nanoTime() - t1) / 1e6);
        System.out.println(out);
    }
}
