
function statusBadgeClasses(status) {
  switch (status) {
    case "queued":
      return "bg-slate-200 text-slate-800";
    case "running":
      return "bg-yellow-100 text-yellow-800";
    case "done":
      return "bg-emerald-100 text-emerald-800";
    case "error":
      return "bg-red-100 text-red-800";
    default:
      return "bg-gray-100 text-gray-800";
  }
}

function resultStatusLabel(item) {
  if (item.status !== "done" || !item.result) return item.status;
  // map API status (OK / UNKNOWN / ERROR) into a nicer label if you like
  return item.result.status || item.status;
}

export function PredictionsTable({ items, loading }) {
  return (
    <section>
      <div className="flex items-center justify-between mb-2">
        <h2 className="text-sm font-semibold text-slate-200">Predictions</h2>
        {loading && (
          <span className="text-xs text-slate-400">
            Processing… this can take time for scanned PDFs.
          </span>
        )}
      </div>

      {items && items.length > 0 ? (
        <div className="overflow-auto max-h-80 rounded-xl border border-slate-800">
          <table className="min-w-full text-xs text-left">
            <thead className="bg-slate-900/80 text-slate-300">
              <tr>
                <th className="px-3 py-2">File</th>
                <th className="px-3 py-2">Status</th>
                <th className="px-3 py-2">Prediction</th>
                <th className="px-3 py-2">Best Label</th>
                <th className="px-3 py-2">Confidence</th>
                <th className="px-3 py-2">Error</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800 bg-slate-900/40">
              {items.map((it) => {
                const r = it.result || {};
                return (
                  <tr key={it.id} className="hover:bg-slate-800/60">
                    <td className="px-3 py-2 text-slate-200 max-w-xs truncate">
                      {it.file.name}
                    </td>
                    <td className="px-3 py-2">
                      <span
                        className={
                          "inline-flex px-2 py-0.5 rounded-full text-[0.65rem] font-semibold " +
                          statusBadgeClasses(it.status)
                        }
                      >
                        {resultStatusLabel(it)}
                      </span>
                    </td>
                    <td className="px-3 py-2 text-slate-100">
                      {r.prediction ?? "-"}
                    </td>
                    <td className="px-3 py-2 text-slate-300">
                      {r.best_label ?? "-"}
                    </td>
                    <td className="px-3 py-2 text-slate-300">
                      {typeof r.confidence === "number"
                        ? r.confidence.toFixed(3)
                        : "-"}
                    </td>
                    <td className="px-3 py-2 text-slate-400 max-w-xs truncate">
                      {it.error || r.error || ""}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      ) : (
        <p className="text-xs text-slate-500">
          No predictions yet. Select some PDFs and click{" "}
          <span className="font-semibold text-slate-300">Predict</span>.
        </p>
      )}
    </section>
  );
}
