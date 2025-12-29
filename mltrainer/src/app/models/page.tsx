"use client";

import { useState, useEffect } from 'react';

interface Model {
    name: string;
    path: string;
    fullPath: string;
    size: number;
    created: string;
}

export default function ModelsPage() {
    const [models, setModels] = useState<Model[]>([]);
    const [loading, setLoading] = useState(true);
    const [selectedModel, setSelectedModel] = useState<Model | null>(null);
    const [servingPort, setServingPort] = useState<number | null>(null);
    const [predictionData, setPredictionData] = useState('');
    const [predictionResult, setPredictionResult] = useState<any>(null);
    const [logs, setLogs] = useState('');

    useEffect(() => {
        fetchModels();
    }, []);

    const fetchModels = async () => {
        try {
            const res = await fetch('/api/models');
            const data = await res.json();
            setModels(data.models || []);
        } catch (err) {
            console.error('Failed to fetch models:', err);
        } finally {
            setLoading(false);
        }
    };

    const handleServe = async (model: Model) => {
        setLogs(`Starting server for ${model.name}...`);
        try {
            const res = await fetch('/api/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    action: 'start',
                    modelPath: model.path,
                    port: 8000
                })
            });

            const data = await res.json();
            if (!res.ok) throw new Error(data.error);

            setServingPort(data.port);
            setSelectedModel(model);
            setLogs(`Server started on port ${data.port}`);
        } catch (err: any) {
            setLogs(`Error starting server: ${err.message}`);
        }
    };

    const handlePredict = async () => {
        if (!servingPort) return;

        try {
            const data = JSON.parse(predictionData);
            // Wrap in array if it's a single object, as API expects list
            const payload = Array.isArray(data) ? data : [data];

            const res = await fetch('/api/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    action: 'predict',
                    port: servingPort,
                    data: payload
                })
            });

            const result = await res.json();
            if (!res.ok) throw new Error(result.error);

            setPredictionResult(result);
        } catch (err: any) {
            setPredictionResult({ error: err.message });
        }
    };

    return (
        <div className="min-h-screen bg-gray-50 dark:bg-zinc-900 p-8">
            <div className="max-w-6xl mx-auto">
                <h1 className="text-3xl font-bold mb-8 text-gray-900 dark:text-white">Manage Models</h1>

                <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
                    {/* Model List */}
                    <div className="lg:col-span-1 bg-white dark:bg-zinc-800 rounded-lg shadow overflow-hidden">
                        <div className="p-4 border-b border-gray-200 dark:border-zinc-700">
                            <h2 className="text-lg font-medium text-gray-900 dark:text-white">Available Models</h2>
                        </div>
                        <div className="divide-y divide-gray-200 dark:divide-zinc-700 max-h-[600px] overflow-y-auto">
                            {loading ? (
                                <div className="p-4 text-center text-gray-500">Loading...</div>
                            ) : models.length === 0 ? (
                                <div className="p-4 text-center text-gray-500">No models found</div>
                            ) : (
                                models.map((model) => (
                                    <div
                                        key={model.name}
                                        className={`p-4 hover:bg-gray-50 dark:hover:bg-zinc-700 cursor-pointer transition-colors ${selectedModel?.name === model.name ? 'bg-blue-50 dark:bg-blue-900/20' : ''
                                            }`}
                                        onClick={() => setSelectedModel(model)}
                                    >
                                        <div className="font-medium text-gray-900 dark:text-white truncate">{model.name}</div>
                                        <div className="text-sm text-gray-500 dark:text-gray-400 mt-1">
                                            {(model.size / 1024).toFixed(1)} KB • {new Date(model.created).toLocaleDateString()}
                                        </div>
                                        <button
                                            onClick={(e) => {
                                                e.stopPropagation();
                                                handleServe(model);
                                            }}
                                            className="mt-3 w-full px-3 py-1.5 text-xs font-medium text-blue-700 bg-blue-100 rounded-md hover:bg-blue-200 dark:text-blue-300 dark:bg-blue-900/50 dark:hover:bg-blue-900"
                                        >
                                            Serve Model
                                        </button>
                                    </div>
                                ))
                            )}
                        </div>
                    </div>

                    {/* Inference Panel */}
                    <div className="lg:col-span-2 space-y-6">
                        {/* Server Status */}
                        <div className="bg-white dark:bg-zinc-800 rounded-lg shadow p-6">
                            <h2 className="text-lg font-medium text-gray-900 dark:text-white mb-4">Server Status</h2>
                            <div className="bg-black rounded p-4 font-mono text-sm text-green-400">
                                {logs || 'No server running. Select a model and click "Serve".'}
                            </div>
                        </div>

                        {/* Prediction Form */}
                        {servingPort && (
                            <div className="bg-white dark:bg-zinc-800 rounded-lg shadow p-6">
                                <h2 className="text-lg font-medium text-gray-900 dark:text-white mb-4">Run Inference</h2>
                                <div className="space-y-4">
                                    <div>
                                        <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                                            Input Data (JSON)
                                        </label>
                                        <textarea
                                            rows={5}
                                            className="w-full px-3 py-2 border border-gray-300 dark:border-zinc-600 rounded-md shadow-sm focus:ring-blue-500 focus:border-blue-500 bg-white dark:bg-zinc-700 text-gray-900 dark:text-white font-mono text-sm"
                                            value={predictionData}
                                            onChange={(e) => setPredictionData(e.target.value)}
                                            placeholder='[{"feature1": 1.0, "feature2": "value"}]'
                                        />
                                    </div>
                                    <button
                                        onClick={handlePredict}
                                        className="px-4 py-2 bg-green-600 text-white rounded-md hover:bg-green-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-green-500"
                                    >
                                        Predict
                                    </button>
                                </div>

                                {predictionResult && (
                                    <div className="mt-6">
                                        <h3 className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">Result</h3>
                                        <pre className="bg-gray-100 dark:bg-zinc-900 p-4 rounded-md overflow-x-auto text-sm">
                                            {JSON.stringify(predictionResult, null, 2)}
                                        </pre>
                                    </div>
                                )}
                            </div>
                        )}
                    </div>
                </div>
            </div>
        </div>
    );
}
