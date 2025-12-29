import Link from "next/link";

export default function Home() {
  return (
    <div className="flex min-h-screen flex-col items-center justify-center bg-gray-50 dark:bg-zinc-900 p-4">
      <div className="max-w-3xl w-full text-center space-y-8">
        <h1 className="text-5xl font-bold tracking-tight text-gray-900 dark:text-white sm:text-6xl">
          MLtrainer <span className="text-blue-600">UI</span>
        </h1>
        <p className="text-lg leading-8 text-gray-600 dark:text-gray-300">
          A powerful interface for the SynthAI framework. Train, tune, and serve your machine learning models with ease.
        </p>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mt-10">
          <Link
            href="/train"
            className="group relative block p-8 bg-white dark:bg-zinc-800 rounded-2xl shadow-sm hover:shadow-md transition-all border border-gray-200 dark:border-zinc-700 text-left"
          >
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-2xl font-semibold text-gray-900 dark:text-white group-hover:text-blue-600 transition-colors">
                Train Model
              </h2>
              <svg className="w-6 h-6 text-gray-400 group-hover:text-blue-600 transition-colors" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
              </svg>
            </div>
            <p className="text-gray-600 dark:text-gray-400">
              Configure and start new training jobs. Support for Random Forest, XGBoost, and more with hyperparameter tuning.
            </p>
          </Link>

          <Link
            href="/models"
            className="group relative block p-8 bg-white dark:bg-zinc-800 rounded-2xl shadow-sm hover:shadow-md transition-all border border-gray-200 dark:border-zinc-700 text-left"
          >
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-2xl font-semibold text-gray-900 dark:text-white group-hover:text-green-600 transition-colors">
                Manage Models
              </h2>
              <svg className="w-6 h-6 text-gray-400 group-hover:text-green-600 transition-colors" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 012-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10" />
              </svg>
            </div>
            <p className="text-gray-600 dark:text-gray-400">
              View trained models, start serving endpoints, and run inference directly from your browser.
            </p>
          </Link>
        </div>
      </div>
    </div>
  );
}
