package stspredictor;

import basemod.BaseMod;
import basemod.interfaces.PostInitializeSubscriber;
import basemod.interfaces.PostRenderSubscriber;
import basemod.interfaces.PostUpdateSubscriber;
import com.badlogic.gdx.Gdx;
import com.badlogic.gdx.Input;
import com.badlogic.gdx.graphics.Color;
import com.badlogic.gdx.graphics.g2d.SpriteBatch;
import com.evacipated.cardcrawl.modthespire.lib.SpireInitializer;
import com.megacrit.cardcrawl.core.Settings;
import com.megacrit.cardcrawl.helpers.FontHelper;

import java.io.File;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.util.Arrays;
import java.util.List;

/**
 * In-game overlay for the sts/bridge/ CommunicationMod integration, plus an
 * autobattle on/off toggle (F9).
 *
 * This mod does NOT talk to CommunicationMod itself, and does not touch the
 * player/combat/game state in any way -- it is purely a renderer plus a
 * single boolean toggle written to a shared file. The real work (reading
 * CommunicationMod's JSON, running our own search, and -- when autobattle is
 * on -- actually sending a play/end command back to CommunicationMod)
 * happens in a completely separate Python process, launched by
 * CommunicationMod per its own config, per sts/bridge/communication_mod.py's
 * docstring. That process and this mod share two small text files at
 * ~/sts_latest_recommendation.txt and ~/sts_autobattle_enabled.txt
 * (Path.home() on the Python side, System.getProperty("user.home") here --
 * same location on any OS, so this works regardless of where either side's
 * code lives on disk):
 *   - communication_mod.py overwrites the first every time CommunicationMod
 *     pushes a new state (atomically -- write-to-temp-then-rename -- so this
 *     mod never reads a half-written file); this mod polls its last-modified
 *     timestamp once a frame (cheap) and only actually re-reads the content
 *     when that timestamp changes.
 *   - THIS mod owns the second: pressing F9 flips a local boolean and writes
 *     the literal text "true"/"false" to it. communication_mod.py reads that
 *     fresh on every state push (see its _autobattle_enabled()) to decide
 *     whether to just recommend or actually act -- see its own module
 *     docstring for the full list of extra safety checks that still apply
 *     even with the toggle on (never acting on a v1 damage-only fallback,
 *     never sending a command not currently listed as legal, etc.). Default
 *     is OFF: if this file has never been written (a fresh install, or
 *     before the key's ever been pressed this session), communication_mod.py
 *     treats it as off, i.e. exactly the advisory-only behavior this mod
 *     had before autobattle existed.
 *
 * Deliberately minimal beyond that: no combat-phase gating on the toggle
 * itself (you can flip it any time, including outside combat, so it's ready
 * before your next fight starts), no per-card confirmation UI -- the
 * overlay's job is just to make it unambiguous, at a glance, whether the
 * mod is currently advisory-only or actually playing your turns for you.
 */
@SpireInitializer
public class STSPredictorMod implements PostInitializeSubscriber, PostUpdateSubscriber, PostRenderSubscriber {

    private static final File LATEST_FILE =
            new File(System.getProperty("user.home"), "sts_latest_recommendation.txt");
    private static final File AUTOBATTLE_FILE =
            new File(System.getProperty("user.home"), "sts_autobattle_enabled.txt");
    private static final int TOGGLE_KEY = Input.Keys.F9;

    private long lastSeenModified = -1L;
    private List<String> cachedLines = Arrays.asList();
    private boolean autobattleEnabled = false;

    public static void initialize() {
        BaseMod.subscribe(new STSPredictorMod());
    }

    @Override
    public void receivePostInitialize() {
        // Starts false regardless of whatever's on disk from a previous
        // session -- autobattle defaulting back to off every time the game
        // (re)starts is the safer failure mode than silently resuming a
        // setting the player might not remember leaving on. Immediately
        // persists that, so communication_mod.py (which may already be
        // running by the time this fires, per CommunicationMod's own
        // startup order) sees the same reset rather than a stale "true"
        // from last session.
        autobattleEnabled = false;
        writeAutobattleState();
    }

    @Override
    public void receivePostUpdate() {
        if (Gdx.input.isKeyJustPressed(TOGGLE_KEY)) {
            autobattleEnabled = !autobattleEnabled;
            writeAutobattleState();
        }
    }

    private void writeAutobattleState() {
        try {
            Files.write(AUTOBATTLE_FILE.toPath(),
                    (autobattleEnabled ? "true" : "false").getBytes(StandardCharsets.UTF_8));
        } catch (IOException e) {
            // Worst case communication_mod.py keeps reading whatever was
            // there before (or nothing, i.e. off) -- never crash the game
            // over a one-line status file.
        }
    }

    @Override
    public void receivePostRender(SpriteBatch sb) {
        refreshIfChanged();

        float x = 24f * Settings.scale;
        float y = Settings.HEIGHT - (24f * Settings.scale);
        float lineHeight = 22f * Settings.scale;

        // Toggle status is always shown (even outside combat, and even with
        // no recommendation yet) -- the one thing this overlay must never
        // leave ambiguous is whether it's currently just watching or
        // actually playing your turns.
        String toggleLine = "STS Predictor -- Autobattle: " + (autobattleEnabled ? "ON" : "OFF") + " (F9)";
        FontHelper.renderFontLeftTopAligned(sb, FontHelper.tipHeaderFont, toggleLine, x, y,
                autobattleEnabled ? Color.RED : Color.YELLOW);
        y -= lineHeight;

        if (cachedLines.isEmpty()) {
            return;
        }
        for (String line : cachedLines) {
            FontHelper.renderFontLeftTopAligned(sb, FontHelper.tipBodyFont, line, x, y, Color.WHITE);
            y -= lineHeight;
        }
    }

    private void refreshIfChanged() {
        if (!LATEST_FILE.exists()) {
            if (!cachedLines.isEmpty()) {
                cachedLines = Arrays.asList();
            }
            return;
        }
        long modified = LATEST_FILE.lastModified();
        if (modified == lastSeenModified) {
            return; // unchanged since last frame -- skip the actual read
        }
        lastSeenModified = modified;
        try {
            String content = new String(Files.readAllBytes(LATEST_FILE.toPath()), StandardCharsets.UTF_8).trim();
            cachedLines = content.isEmpty() ? Arrays.asList() : Arrays.asList(content.split("\n"));
        } catch (IOException e) {
            // A transient read failure (e.g. caught mid-rename on some
            // filesystems) shouldn't spam the overlay with an error --
            // just keep showing the last good content until the next
            // successful read.
        }
    }
}
