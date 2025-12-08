import { useState } from "react";
import { Toolbar } from "./components/Toolbar";
import { FileUploadZone } from "./components/FileUploadZone";
import { PredictionsTable } from "./components/PredictionsTable";

const API_BASE = "http://127.0.0.1:8000"; // or "http://localhost:8000"

function App() {
  const [items, setItems] = useState([]); // [{ file, id, status, result, error }]
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const handleFilesSelected = (files) => {
    const withState = files.map((f, idx) => ({
      file: f,
      id: `${f.name}-${f.size}-${idx}-${Date.now()}`,
      status: "queued", // queued | running | done | error
      result: null,
      error: "",
    }));
    setItems(withState);
    setError("");
  };

  const updateItem = (id, patch) => {
    setItems((prev) =>
      prev.map((it) => (it.id === id ? { ...it, ...patch } : it))
    );
  };

  const handlePredict = async () => {
    if (items.length === 0) {
      setError("Please select at least one PDF file.");
      return;
    }

    setLoading(true);
    setError("");

    const queue = [...items];

    const runOne = async (item) => {
      updateItem(item.id, { status: "running", error: "", result: null });
      try {
        const formData = new FormData();
        formData.append("file", item.file);

        const res = await fetch(`${API_BASE}/predict`, {
          method: "POST",
          body: formData,
        });

        if (!res.ok) {
          const text = await res.text();
          throw new Error(`API error (${res.status}): ${text}`);
        }

        const data = await res.json(); // PredictionResult
        updateItem(item.id, { status: "done", result: data });
      } catch (err) {
        console.error(err);
        updateItem(item.id, {
          status: "error",
          error: err.message || "Prediction failed.",
        });
      }
    };

    const MAX_PARALLEL = 2;
    let running = 0;
    let index = 0;

    await new Promise((resolve) => {
      const launchNext = () => {
        if (index >= queue.length && running === 0) {
          resolve();
          return;
        }
        while (running < MAX_PARALLEL && index < queue.length) {
          const item = queue[index++];
          running += 1;
          runOne(item).finally(() => {
            running -= 1;
            launchNext();
          });
        }
      };
      launchNext();
    });

    setLoading(false);
  };

  const handleDownloadCsv = async () => {
    if (items.length === 0) {
      setError("Please select at least one PDF file.");
      return;
    }

    setLoading(true);
    setError("");

    try {
      const formData = new FormData();
      items.forEach((it) => formData.append("files", it.file));

      const res = await fetch(`${API_BASE}/predict-batch-csv`, {
        method: "POST",
        body: formData,
      });

      if (!res.ok) {
        const text = await res.text();
        throw new Error(`API error (${res.status}): ${text}`);
      }

      const blob = await res.blob();
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = "predictions_report.csv";
      document.body.appendChild(a);
      a.click();
      a.remove();
      window.URL.revokeObjectURL(url);
    } catch (err) {
      console.error(err);
      setError(err.message || "CSV download failed.");
    } finally {
      setLoading(false);
    }
  };

  const doneCount = items.filter((it) => it.status === "done").length;

  return (
    <div className="min-h-screen bg-slate-900 text-slate-100 flex items-center justify-center px-4 py-8">
      <div className="w-full max-w-5xl bg-slate-950/60 border border-slate-800 rounded-2xl shadow-xl p-6 md:p-8">
        <Toolbar
          loading={loading}
          hasFiles={items.length > 0}
          onPredict={handlePredict}
          onDownloadCsv={handleDownloadCsv}
        />

        <FileUploadZone
          items={items}
          onFilesSelected={handleFilesSelected}
          error={error}
        />

        {items.length > 0 && (
          <div className="mb-2 text-xs text-slate-400">
            Progress: {doneCount}/{items.length} completed
          </div>
        )}

        <PredictionsTable items={items} loading={loading} />
      </div>
    </div>
  );
}

export default App;
