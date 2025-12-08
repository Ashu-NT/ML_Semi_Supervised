
export function FileUploadZone({ items, onFilesSelected, error }) {
  const handleChange = (e) => {
    const files = Array.from(e.target.files || []);
    onFilesSelected(files);
  };

  const files = items.map((it) => it.file);

  return (
    <section className="mb-6">
      <label
        htmlFor="file-input"
        className="flex flex-col items-center justify-center w-full h-32 border-2 border-dashed border-slate-700 rounded-2xl cursor-pointer bg-slate-900/60 hover:bg-slate-900 transition"
      >
        <span className="text-sm text-slate-300">
          Click to select PDFs or drag &amp; drop (multiple allowed)
        </span>
        <span className="text-xs text-slate-500 mt-1">
          Only <code>.pdf</code> files will work with the API
        </span>
        <input
          id="file-input"
          type="file"
          accept="application/pdf"
          multiple
          className="hidden"
          onChange={handleChange}
        />
      </label>

      {files.length > 0 && (
        <div className="mt-3 text-xs text-slate-300">
          <span className="font-medium">{files.length}</span>{" "}
          file{files.length > 1 ? "s" : ""} selected:
          <ul className="mt-1 max-h-24 overflow-auto space-y-0.5">
            {files.map((f, idx) => (
              <li key={`${f.name}-${idx}`} className="truncate text-slate-400">
                • {f.name}
              </li>
            ))}
          </ul>
        </div>
      )}

      {error && (
        <div className="mt-3 rounded-xl border border-red-500/40 bg-red-500/10 text-red-200 text-sm px-3 py-2">
          {error}
        </div>
      )}
    </section>
  );
}
