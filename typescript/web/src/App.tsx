import React, { useState, useEffect } from 'react';
import { OpenAHI, GenerationConfig, ModelInfo } from '@openahi/sdk';
import './App.css';

function App() {
  const [client, setClient] = useState<OpenAHI | null>(null);
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [selectedModel, setSelectedModel] = useState<string>('');
  const [prompt, setPrompt] = useState<string>('');
  const [result, setResult] = useState<string>('');
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [runtimeInfo, setRuntimeInfo] = useState<any>(null);

  useEffect(() => {
    // Initialize client
    const newClient = new OpenAHI({
      baseUrl: 'http://localhost:8080',
      debug: true,
    });
    setClient(newClient);

    // Check runtime and list models
    checkRuntime();
  }, []);

  const checkRuntime = async () => {
    try {
      if (!client) return;

      // Check runtime
      const health = await client.checkRuntime();
      console.log('Runtime health:', health);

      // Get runtime info
      const info = await client.getRuntimeInfo();
      setRuntimeInfo(info);

      // List models
      const modelList = await client.listModels();
      setModels(modelList.models);
      
      // Set default model if available
      if (modelList.models.length > 0) {
        setSelectedModel(modelList.models[0].name);
      }
    } catch (err) {
      console.error('Failed to connect to runtime:', err);
      setError('Failed to connect to OpenAHI runtime. Please ensure it is running.');
    }
  };

  const handleGenerate = async () => {
    if (!client || !prompt.trim()) return;

    setLoading(true);
    setError(null);
    setResult('');

    try {
      const config: GenerationConfig = {
        maxTokens: 100,
        temperature: 0.7,
      };

      const generationResult = await client.generate(prompt, config);
      setResult(generationResult.text);
    } catch (err) {
      console.error('Generation error:', err);
      setError(`Generation failed: ${err}`);
    } finally {
      setLoading(false);
    }
  };

  const handleLoadModel = async (modelName: string) => {
    if (!client) return;

    try {
      await client.loadModel(modelName);
      setError(null);
    } catch (err) {
      console.error('Failed to load model:', err);
      setError(`Failed to load model: ${err}`);
    }
  };

  return (
    <div className="app">
      <header className="header">
        <h1>OpenAHI</h1>
        <p>Open Artificial Hyper Intelligence</p>
      </header>

      <main className="main">
        {error && (
          <div className="error">
            <p>{error}</p>
            <button onClick={checkRuntime}>Retry Connection</button>
          </div>
        )}

        <section className="runtime-info">
          <h2>Runtime</h2>
          {runtimeInfo ? (
            <div>
              <p>Version: {runtimeInfo.version}</p>
              <p>Loaded Models: {runtimeInfo.loadedModels}</p>
              <p>Uptime: {runtimeInfo.uptime}ms</p>
            </div>
          ) : (
            <p>Connecting to runtime...</p>
          )}
        </section>

        <section className="models">
          <h2>Models</h2>
          <div className="model-list">
            {models.map((model) => (
              <div
                key={`${model.name}@${model.version}`}
                className={`model-item ${selectedModel === model.name ? 'selected' : ''}`}
                onClick={() => setSelectedModel(model.name)}
              >
                <h3>{model.name}</h3>
                <p>{model.version}</p>
                <button onClick={(e) => {
                  e.stopPropagation();
                  handleLoadModel(model.name);
                }}>
                  Load
                </button>
              </div>
            ))}
          </div>
        </section>

        <section className="generation">
          <h2>Text Generation</h2>
          <div className="generation-form">
            <select
              value={selectedModel}
              onChange={(e) => setSelectedModel(e.target.value)}
            >
              {models.map((model) => (
                <option key={`${model.name}@${model.version}`} value={model.name}>
                  {model.name}@{model.version}
                </option>
              ))}
            </select>

            <textarea
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              placeholder="Enter your prompt..."
            />

            <button
              onClick={handleGenerate}
              disabled={loading || !prompt.trim()}
            >
              {loading ? 'Generating...' : 'Generate'}
            </button>
          </div>

          {result && (
            <div className="result">
              <h3>Result:</h3>
              <pre>{result}</pre>
            </div>
          )}
        </section>
      </main>

      <footer className="footer">
        <p>OpenAHI is created by ZoDev</p>
        <p>Composter 1.00.0 is the first OpenAHI model</p>
      </footer>
    </div>
  );
}

export default App;
