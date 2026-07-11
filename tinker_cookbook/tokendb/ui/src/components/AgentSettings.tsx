// Chat agent settings: provider / model / API key form (POST
// /api/agent/config; the key lives in server memory only), plus the header
// gear button that opens the same form in a popover.

import { useState } from "react";
import { getJSON, postJSON, type AgentConfig } from "../api";
import { useApi } from "../hooks/useApi";

const CUSTOM = "__custom__";

export function AgentSettings({ onSaved }: { onSaved?: () => void }) {
  const config = useApi(() => getJSON<AgentConfig>("/api/agent/config"), []);
  const [provider, setProvider] = useState<string | null>(null); // null = unchanged
  const [modelChoice, setModelChoice] = useState<string | null>(null); // null = unchanged
  const [customModel, setCustomModel] = useState("");
  const [apiKey, setApiKey] = useState("");
  const [saving, setSaving] = useState(false);
  const [saved, setSaved] = useState(false);
  const [error, setError] = useState("");
  const current = config.data;

  const activeProvider = provider ?? current?.provider ?? "anthropic";
  const knownModels = current?.models?.[activeProvider] ?? [];
  // Untouched dropdown reflects the server's current model; a configured
  // model outside the curated list shows as "custom".
  const selectedModel =
    modelChoice ??
    (current === null || knownModels.includes(current.model) ? current?.model ?? "" : CUSTOM);
  const customValue =
    modelChoice === null && selectedModel === CUSTOM ? current?.model ?? "" : customModel;

  const changeProvider = (next: string) => {
    setProvider(next);
    // Each provider gets its own default preselected.
    setModelChoice(current?.default_model?.[next] ?? null);
  };

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
        </select>
      </label>
      <label>
        <span>model</span>
        <select value={selectedModel} onChange={(event) => setModelChoice(event.target.value)}>
          {knownModels.map((model) => (
            <option key={model} value={model}>
              {model}
            </option>
          ))}
          <option value={CUSTOM}>Custom…</option>
        </select>
      </label>
      {selectedModel === CUSTOM && (
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
        <code>ANTHROPIC_API_KEY</code> / <code>OPENAI_API_KEY</code> in the server environment
        also works.
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
