

export function Toolbar({ loading, hasFiles, onPredict, onDownloadCsv }) {
  return (
    <header className="flex flex-col md:flex-row md:items-end md:justify-between gap-4 mb-6">
      <div>
        <h1 className="text-2xl md:text-3xl font-semibold text-white">
          Document Classifier UI
        </h1>
        <p className="text-slate-400 text-sm mt-1">
          Upload PDFs, classify them via AI, and export a CSV report.
        </p>
      </div>
      <div className="flex gap-3">
        <button
          type="button"
          onClick={onPredict}
          disabled={loading || !hasFiles}
          className="inline-flex items-center justify-center rounded-xl px-4 py-2 text-sm font-medium bg-emerald-500 hover:bg-emerald-400 disabled:bg-emerald-800 disabled:text-emerald-200 transition"
        >
          {loading ? "Running..." : "Predict (JSON)"}
        </button>
        <button
          type="button"
          onClick={onDownloadCsv}
          disabled={loading || !hasFiles}
          className="inline-flex items-center justify-center rounded-xl px-4 py-2 text-sm font-medium bg-sky-500 hover:bg-sky-400 disabled:bg-sky-800 disabled:text-sky-200 transition"
        >
          Download CSV
        </button>
      </div>
    </header>
  );
}
