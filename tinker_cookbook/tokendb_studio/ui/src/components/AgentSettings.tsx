// Chat agent settings: provider / model / API key form (POST
// /api/agent/config; the key lives in server memory only), plus the header
// gear button that opens the same form in a popover.

import { useEffect, useState } from "react";
import { getJSON, postJSON, type AgentConfig } from "../api";
import { useApi } from "../hooks/useApi";

const CUSTOM = "__custom__";

export function AgentSettings({ onSaved }: { onSaved?: () => void }) {
  const [provider, setProvider] = useState<string | null>(null); // null = unchanged
  // Selecting tinker asks the server to (lazily) fetch its supported-model
  // list from server capabilities; other providers use the static config.
  const config = useApi(
    () =>
      getJSON<AgentConfig>(
        `/api/agent/config${provider === "tinker" ? "?provider=tinker" : ""}`,
      ),
    [provider],
  );
  const [modelChoice, setModelChoice] = useState<string | null>(null); // null = unchanged
  const [customModel, setCustomModel] = useState("");
  const [apiKey, setApiKey] = useState("");
  const [saving, setSaving] = useState(false);
  const [saved, setSaved] = useState(false);
  const [error, setError] = useState("");
  const current = config.data;

  const activeProvider = provider ?? current?.provider ?? "anthropic";
  const knownModels = current?.models?.[activeProvider] ?? [];
  const tinkerActive = activeProvider === "tinker";
  const modelsLoading = tinkerActive && config.loading;
  // Untouched dropdown reflects the server's current model; a configured
  // model outside the curated list shows as "custom".
  const selectedModel =
    modelChoice ??
    (current === null || knownModels.includes(current.model) ? current?.model ?? "" : CUSTOM);
  const customValue =
    modelChoice === null && selectedModel === CUSTOM ? current?.model ?? "" : customModel;

  const changeProvider = (next: string) => {
    setProvider(next);
    // Each provider gets its own default preselected. For tinker the real
    // default arrives with the fetched model list (effect below).
    setModelChoice(current?.default_model?.[next] ?? null);
  };

  // When the tinker model list arrives, snap the selection to the fetched
  // default unless the user already picked a model that exists in the list
  // (or the server's configured model does).
  useEffect(() => {
    if (activeProvider !== "tinker" || !current) return;
    const models = current.models?.tinker ?? [];
    if (models.length === 0) return;
    setModelChoice((choice) => {
      if (choice === CUSTOM || (choice !== null && models.includes(choice))) return choice;
      if (choice === null && models.includes(current.model)) return null;
      return current.default_model?.tinker ?? models[0];
    });
  }, [activeProvider, current]);

  const save = async () => {
    setSaving(true);
    setError("");
    try {
      const body: Record<string, string> = {};
      if (provider !== null) body.provider = provider;
      if (modelChoice !== null) {
        body.model = modelChoice === CUSTOM ? customModel.trim() : modelChoice;
      }
      if (apiKey) body.api_key = apiKey;
      await postJSON<AgentConfig>("/api/agent/config", body);
      setApiKey("");
      setSaved(true);
      setTimeout(() => setSaved(false), 1500);
      config.reload();
      onSaved?.();
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setSaving(false);
    }
  };

  return (
    <div className="agent-settings">
      <label>
        <span>provider</span>
        <select value={activeProvider} onChange={(event) => changeProvider(event.target.value)}>
          <option value="anthropic">anthropic</option>
          <option value="openai">openai</option>
          <option value="tinker">tinker</option>
        </select>
      </label>
      <label>
        <span>model</span>
        <select
          value={modelsLoading ? "" : selectedModel}
          disabled={modelsLoading}
          onChange={(event) => setModelChoice(event.target.value)}
        >
          {modelsLoading && <option value="">loading models…</option>}
          {knownModels.map((model) => (
            <option key={model} value={model}>
              {model}
            </option>
          ))}
          <option value={CUSTOM}>Custom…</option>
        </select>
      </label>
      {tinkerActive && !modelsLoading && knownModels.length === 0 && (
        <p className="muted small">
          No models available: set <code>TINKER_API_KEY</code> in the server environment (or paste
          a key below and save), then reopen settings.
          {current?.tinker_models_error ? ` (${current.tinker_models_error})` : ""}
        </p>
      )}
      {selectedModel === CUSTOM && !modelsLoading && (
        <label>
          <span>custom model</span>
          <input
            placeholder="model id"
            value={customValue}
            onChange={(event) => {
              setModelChoice(CUSTOM);
              setCustomModel(event.target.value);
            }}
          />
        </label>
      )}
      <label>
        <span>API key</span>
        <input
          type="password"
          placeholder={current?.has_key ? "configured" : "paste a key"}
          value={apiKey}
          onChange={(event) => setApiKey(event.target.value)}
        />
      </label>
      <button className="primary" disabled={saving} onClick={() => void save()}>
        {saved ? "saved" : "Save"}
      </button>
      {error && <div className="error small">{error}</div>}
      <p className="muted small">
        The key is held in server memory only; it is never written to disk. Setting{" "}
        <code>ANTHROPIC_API_KEY</code> / <code>OPENAI_API_KEY</code> / <code>TINKER_API_KEY</code>{" "}
        in the server environment also works. The tinker provider chats with any model served by
        Tinker; its model list comes from the service's capabilities.
      </p>
    </div>
  );
}

/** Small gear button for the app header; toggles the settings popover. */
export function SettingsButton() {
  const [open, setOpen] = useState(false);
  return (
    <span className="settings-wrap">
      <button title="chat agent settings" aria-label="chat agent settings" onClick={() => setOpen((o) => !o)}>
        {"⚙︎"}
      </button>
      {open && (
        <div className="settings-pop">
          <AgentSettings />
        </div>
      )}
    </span>
  );
}
