import type { ReactNode } from "react";

export interface TextFilterField {
  name: string;
  placeholder: string;
  width?: string;
}

/**
 * Row of filter inputs. Text fields are declared via `fields`; selects,
 * toggles, and buttons come in as children after them.
 */
export function FilterBar({
  fields,
  values,
  onChange,
  onSubmit,
  children,
}: {
  fields: TextFilterField[];
  values: Record<string, string>;
  onChange: (name: string, value: string) => void;
  onSubmit?: () => void;
  children?: ReactNode;
}) {
  return (
    <div
      className="filter-bar"
      onKeyDown={(event) => {
        if (event.key === "Enter") onSubmit?.();
      }}
    >
      {fields.map(({ name, placeholder, width }) => (
        <input
          key={name}
          placeholder={placeholder}
          style={width ? { width } : undefined}
          value={values[name] ?? ""}
          onChange={(event) => onChange(name, event.target.value)}
        />
      ))}
      {children}
    </div>
  );
}
