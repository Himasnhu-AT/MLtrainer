"use client";

import { useState } from 'react';

export default function TrainPage() {
    const [formData, setFormData] = useState({
        data: '',
        schema: '',
        modelType: 'random_forest',
        output: 'models',
        tune: false,
        iterations: 10
    });
    const [loading, setLoading] = useState(false);
    const [logs, setLogs] = useState('');
    const [error, setError] = useState('');

    const handleSubmit = async (e: React.FormEvent) => {
        e.preventDefault();
        setLoading(true);
        setError('');
        setLogs('Starting training job...');

        try {
            const res = await fetch('/api/train', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(formData)
            });

            const result = await res.json();

            if (!res.ok) {
                throw new Error(result.error || 'Training failed');
            }

            setLogs(result.stdout || 'Training completed successfully.');
        } catch (err: any) {
            setError(err.message);
            setLogs(prev => prev + '\nTraining failed.');
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="min-h-screen bg-gray-50 dark:bg-zinc-900 p-8">
            <div className="max-w-4xl mx-auto">
                <h1 className="text-3xl font-bold mb-8 text-gray-900 dark:text-white">Train New Model</h1>

                <div className="bg-white dark:bg-zinc-800 rounded-lg shadow p-6 mb-8">
                    <form onSubmit={handleSubmit} className="space-y-6">
                        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                            <div>
                                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                                    Data Path (CSV)
                                </label>
                                <input
                                    type="text"
                                    required
                                    className="w-full px-3 py-2 border border-gray-300 dark:border-zinc-600 rounded-md shadow-sm focus:ring-blue-500 focus:border-blue-500 bg-white dark:bg-zinc-700 text-gray-900 dark:text-white"
                                    value={formData.data}
                                    onChange={e => setFormData({ ...formData, data: e.target.value })}
                                    placeholder="e.g., synthai/data/raw/data.csv"
                                />
                            </div>

                            <div>
                                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                                    Schema Path (JSON)
                                </label>
                                <input
                                    type="text"
                                    required
                                    className="w-full px-3 py-2 border border-gray-300 dark:border-zinc-600 rounded-md shadow-sm focus:ring-blue-500 focus:border-blue-500 bg-white dark:bg-zinc-700 text-gray-900 dark:text-white"
                                    value={formData.schema}
                                    onChange={e => setFormData({ ...formData, schema: e.target.value })}
                                    placeholder="e.g., synthai/config/schema.json"
                                />
                            </div>

                            <div>
                                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                                    Model Type
                                </label>
                                <select
                                    className="w-full px-3 py-2 border border-gray-300 dark:border-zinc-600 rounded-md shadow-sm focus:ring-blue-500 focus:border-blue-500 bg-white dark:bg-zinc-700 text-gray-900 dark:text-white"
                                    value={formData.modelType}
                                    onChange={e => setFormData({ ...formData, modelType: e.target.value })}
                                >
                                    <option value="random_forest">Random Forest</option>
                                    <option value="xgboost">XGBoost</option>
                                    <option value="logistic_regression">Logistic Regression</option>
                                    <option value="linear_regression">Linear Regression</option>
                                </select>
                            </div>

                            <div>
                                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                                    Output Directory
                                </label>
                                <input
                                    type="text"
                                    className="w-full px-3 py-2 border border-gray-300 dark:border-zinc-600 rounded-md shadow-sm focus:ring-blue-500 focus:border-blue-500 bg-white dark:bg-zinc-700 text-gray-900 dark:text-white"
                                    value={formData.output}
                                    onChange={e => setFormData({ ...formData, output: e.target.value })}
                                />
                            </div>
                        </div>

                        <div className="flex items-center space-x-4">
                            <div className="flex items-center">
                                <input
                                    type="checkbox"
                                    id="tune"
                                    className="h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300 rounded"
                                    checked={formData.tune}
                                    onChange={e => setFormData({ ...formData, tune: e.target.checked })}
                                />
                                <label htmlFor="tune" className="ml-2 block text-sm text-gray-900 dark:text-gray-300">
                                    Enable Hyperparameter Tuning
                                </label>
                            </div>

                            {formData.tune && (
                                <div className="flex items-center space-x-2">
                                    <label className="text-sm text-gray-700 dark:text-gray-300">Iterations:</label>
                                    <input
                                        type="number"
                                        min="1"
                                        className="w-20 px-2 py-1 border border-gray-300 dark:border-zinc-600 rounded-md shadow-sm focus:ring-blue-500 focus:border-blue-500 bg-white dark:bg-zinc-700 text-gray-900 dark:text-white"
                                        value={formData.iterations}
                                        onChange={e => setFormData({ ...formData, iterations: parseInt(e.target.value) })}
                                    />
                                </div>
                            )}
                        </div>

                        <div className="flex justify-end">
                            <button
                                type="submit"
                                disabled={loading}
                                className={`px-4 py-2 border border-transparent rounded-md shadow-sm text-sm font-medium text-white bg-blue-600 hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500 ${loading ? 'opacity-50 cursor-not-allowed' : ''
                                    }`}
                            >
                                {loading ? 'Training...' : 'Start Training'}
                            </button>
                        </div>
                    </form>
                </div>

                {error && (
                    <div className="bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-md p-4 mb-8">
                        <div className="flex">
                            <div className="flex-shrink-0">
                                <svg className="h-5 w-5 text-red-400" viewBox="0 0 20 20" fill="currentColor">
                                    <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clipRule="evenodd" />
                                </svg>
                            </div>
                            <div className="ml-3">
                                <h3 className="text-sm font-medium text-red-800 dark:text-red-200">Error</h3>
                                <div className="mt-2 text-sm text-red-700 dark:text-red-300">{error}</div>
                            </div>
                        </div>
                    </div>
                )}

                {logs && (
                    <div className="bg-zinc-900 rounded-lg shadow p-6">
                        <h3 className="text-lg font-medium text-white mb-4">Training Logs</h3>
                        <pre className="bg-black p-4 rounded text-green-400 font-mono text-sm overflow-x-auto whitespace-pre-wrap">
                            {logs}
                        </pre>
                    </div>
                )}
            </div>
        </div>
    );
}
