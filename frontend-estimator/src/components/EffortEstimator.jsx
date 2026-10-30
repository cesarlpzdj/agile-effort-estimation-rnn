import { useState } from 'react';

// Catálogo de los 16 proyectos del benchmark Deep-SE con sus medianas históricas
// obtenidas de todos los tickets previos al periodo de prueba (80% train + val, idéntico al artículo)
const PROJECTS = [
  { id: 'appceleratorstudio', name: 'Appcelerator Studio', median: 5.0 },
  { id: 'aptanastudio', name: 'Aptana Studio', median: 8.0 },
  { id: 'bamboo', name: 'Bamboo (Atlassian)', median: 2.0 },
  { id: 'clover', name: 'Clover (Atlassian)', median: 2.0 },
  { id: 'datamanagement', name: 'Data Management', median: 4.0 },
  { id: 'duracloud', name: 'DuraCloud', median: 1.0 },
  { id: 'jirasoftware', name: 'Jira Software (Atlassian)', median: 3.0 },
  { id: 'mesos', name: 'Apache Mesos', median: 3.0 },
  { id: 'moodle', name: 'Moodle LMS', median: 8.0 },
  { id: 'mule', name: 'Mule ESB', median: 5.0 },
  { id: 'mulestudio', name: 'Mule Studio', median: 5.0 },
  { id: 'springxd', name: 'Spring XD', median: 3.0 },
  { id: 'talenddataquality', name: 'Talend Data Quality', median: 5.0 },
  { id: 'talendesb', name: 'Talend ESB', median: 2.0 },
  { id: 'titanium', name: 'Titanium Mobile SDK', median: 5.0 },
  { id: 'usergrid', name: 'Apache Usergrid', median: 3.0 },
];

const PRESET_EXAMPLES = [
  {
    label: 'Moodle (Bug SSO)',
    ticket_id: 'MDL-4821',
    project_domain: 'moodle',
    issue_type: 'Bug',
    title: 'Unhandled exception in SSO authentication module',
    description: 'When a user attempts single sign-on authentication through SAML2 provider, the database session lock times out after 30 seconds. This causes an unhandled fatal error and prevents student dashboard redirection.',
  },
  {
    label: 'Mesos (Story)',
    ticket_id: 'MESOS-3190',
    project_domain: 'mesos',
    issue_type: 'Story',
    title: 'Support dynamic reservation allocator for slave agent resources',
    description: 'Frameworks should be able to dynamically reserve CPU and memory resources on a slave node without restarting the agent daemon. Requires protocol buffer changes and master allocator state validation.',
  },
  {
    label: 'DuraCloud (Task)',
    ticket_id: 'DURACLOUD-902',
    project_domain: 'duracloud',
    issue_type: 'Task',
    title: 'Exponential backoff for Amazon S3 sync connection resets',
    description: 'Implement exponential backoff retry mechanism with jitter when transferring large assets to S3 buckets during transient network disconnects.',
  },
  {
    label: 'Clover (Bug Parser)',
    ticket_id: 'CLOV-1420',
    project_domain: 'clover',
    issue_type: 'Bug',
    title: 'Clover instrumentation fails with record syntax on Java 17',
    description: 'The source code parser throws ParseException when analyzing Java 17 record classes. The AST visitor needs to skip compact constructors and record canonical headers.',
  },
];

export default function EffortEstimator() {
  const [formData, setFormData] = useState({
    ticket_id: 'PRJ-8492',
    project_domain: 'moodle',
    title: 'Unhandled exception in the authentication module',
    description: 'When a user tries to log in using the SSO provider, the database connection times out after 30 seconds. This needs to be caught and handled gracefully.',
    issue_type: 'Bug',
  });

  const [isLoading, setIsLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  const selectedProject = PROJECTS.find(p => p.id === formData.project_domain);

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData(prev => ({ ...prev, [name]: value }));
  };

  const handleApplyExample = (example) => {
    setFormData({
      ticket_id: example.ticket_id,
      project_domain: example.project_domain,
      issue_type: example.issue_type,
      title: example.title,
      description: example.description,
    });
    setResult(null);
    setError(null);
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setIsLoading(true);
    setError(null);
    setResult(null);

    try {
      const response = await fetch('http://127.0.0.1:8000/api/v1/predict', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(formData),
      });

      if (!response.ok) {
        throw new Error(`El servidor respondió con código HTTP ${response.status}`);
      }

      const data = await response.json();
      setResult(data);
    } catch (err) {
      setError(
        err.message.includes('Failed to fetch') || err.message.includes('NetworkError')
          ? 'No se pudo conectar con el microservicio FastAPI en http://127.0.0.1:8000. Asegúrese de que el backend esté ejecutándose (uvicorn api:app --reload).'
          : err.message
      );
    } finally {
      setIsLoading(false);
    }
  };

  // Comparación analítica de soporte a la decisión
  const renderDeliberationInsight = () => {
    if (!result?.prediction) return null;

    const predRounded = result.prediction.story_points_rounded;
    const baseline = result.prediction.project_median_baseline;
    const diff = Math.abs(predRounded - baseline);

    if (diff === 0) {
      return (
        <div className="bg-emerald-50 border border-emerald-300 rounded-lg p-4 text-emerald-900 text-sm">
          <div className="flex items-center gap-2 font-semibold text-emerald-800">
            <span className="text-lg">✓</span>
            <span>Concordancia Metodológica Total</span>
          </div>
          <p className="mt-1">
            La red neuronal y la mediana histórica del proyecto coinciden exactamente en <strong>{predRounded} Story Points</strong>.
            Existe alta estabilidad estadística para fijar este valor en el Sprint.
          </p>
        </div>
      );
    }

    if (diff <= 2) {
      return (
        <div className="bg-blue-50 border border-blue-300 rounded-lg p-4 text-blue-900 text-sm">
          <div className="flex items-center gap-2 font-semibold text-blue-800">
            <span className="text-lg">ℹ</span>
            <span>Consenso Cercano (Diferencia de {diff} SP)</span>
          </div>
          <p className="mt-1">
            La predicción neuronal (<strong>{predRounded} SP</strong>) se sitúa cerca de la mediana típica del dominio (<strong>{baseline} SP</strong>).
            Se aconseja usar <strong>{predRounded} SP</strong> como referencia inicial en el Planning Poker.
          </p>
        </div>
      );
    }

    return (
      <div className="bg-amber-50 border border-amber-300 rounded-lg p-4 text-amber-900 text-sm">
        <div className="flex items-center gap-2 font-semibold text-amber-800">
          <span className="text-lg">⚠</span>
          <span>Señal de Revisión por Divergencia (Diferencia de {diff} SP)</span>
        </div>
        <p className="mt-1">
          La estimación de la red neuronal (<strong>{predRounded} SP</strong>) difiere de la mediana histórica del proyecto (<strong>{baseline} SP</strong>).
          Esta discrepancia no implica un fallo del requerimiento, sino una señal para revisar: puede originarse en atipicidad del ticket, complejidad no capturada o diferencias en el dimensionamiento. Se aconseja una breve deliberación en el Planning Poker antes de comprometer la tarea.
        </p>
      </div>
    );
  };

  return (
    <div className="max-w-4xl mx-auto my-8 p-6 md:p-8 bg-white rounded-2xl shadow-lg border border-gray-100 font-sans">
      {/* Encabezado */}
      <div className="border-b border-gray-200 pb-5 mb-6">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div>
            <span className="text-xs font-bold uppercase tracking-wider text-blue-600 bg-blue-50 px-2.5 py-1 rounded-md">
              Deep-SE Decision Support
            </span>
            <h1 className="text-2xl md:text-3xl font-extrabold text-gray-900 mt-2">
              Estimador de Esfuerzo Ágil
            </h1>
          </div>
          <div className="text-right text-xs text-gray-500">
            <p className="font-semibold text-gray-700">Modelo: LSTM v2 Secuencial</p>
            <p>Validación Cronológica Leak-Free</p>
          </div>
        </div>
        <p className="text-sm text-gray-600 mt-2 leading-relaxed">
          Asistente predictivo para estimación de Story Points basado en procesamiento de lenguaje natural y contraste con la mediana histórica del proyecto.
        </p>

        {/* Botones de Ejemplos Rápidos */}
        <div className="mt-4 pt-3 border-t border-gray-100">
          <span className="text-xs font-medium text-gray-500 mr-2">Cargar ejemplo de prueba:</span>
          <div className="inline-flex flex-wrap gap-2 mt-1">
            {PRESET_EXAMPLES.map((ex) => (
              <button
                key={ex.label}
                type="button"
                onClick={() => handleApplyExample(ex)}
                className="text-xs bg-gray-100 hover:bg-blue-50 hover:text-blue-700 text-gray-700 px-3 py-1 rounded-full border border-gray-200 transition"
              >
                {ex.label}
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Formulario */}
      <form onSubmit={handleSubmit} className="space-y-5">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div>
            <label className="block text-xs font-semibold uppercase tracking-wider text-gray-600 mb-1">
              ID del Ticket
            </label>
            <input
              type="text"
              name="ticket_id"
              value={formData.ticket_id}
              onChange={handleChange}
              required
              className="w-full px-3 py-2 text-sm bg-gray-50 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:bg-white transition"
              placeholder="Ej. MDL-1024"
            />
          </div>

          <div>
            <label className="block text-xs font-semibold uppercase tracking-wider text-gray-600 mb-1">
              Ecosistema / Proyecto (16 Deep-SE)
            </label>
            <select
              name="project_domain"
              value={formData.project_domain}
              onChange={handleChange}
              className="w-full px-3 py-2 text-sm bg-gray-50 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:bg-white transition"
            >
              {PROJECTS.map((proj) => (
                <option key={proj.id} value={proj.id}>
                  {proj.name} (Mediana: {proj.median} SP)
                </option>
              ))}
            </select>
          </div>

          <div>
            <label className="block text-xs font-semibold uppercase tracking-wider text-gray-600 mb-1">
              Tipo de Incidencia
            </label>
            <select
              name="issue_type"
              value={formData.issue_type}
              onChange={handleChange}
              className="w-full px-3 py-2 text-sm bg-gray-50 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:bg-white transition"
            >
              <option value="Bug">Bug (Defecto)</option>
              <option value="Story">User Story (Historia de Usuario)</option>
              <option value="Task">Task (Tarea Técnica)</option>
              <option value="Improvement">Improvement (Mejora)</option>
              <option value="New Feature">New Feature (Nueva Característica)</option>
            </select>
          </div>
        </div>

        <div>
          <label className="block text-xs font-semibold uppercase tracking-wider text-gray-600 mb-1">
            Título del Requerimiento
          </label>
          <input
            type="text"
            name="title"
            value={formData.title}
            onChange={handleChange}
            required
            className="w-full px-3 py-2 text-sm bg-gray-50 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:bg-white transition"
            placeholder="Título conciso del requerimiento o bug..."
          />
        </div>

        <div>
          <label className="block text-xs font-semibold uppercase tracking-wider text-gray-600 mb-1">
            Descripción Detallada
          </label>
          <textarea
            name="description"
            value={formData.description}
            onChange={handleChange}
            required
            rows="4"
            className="w-full px-3 py-2 text-sm bg-gray-50 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:bg-white transition"
            placeholder="Describe el comportamiento esperado, dependencias y condiciones de aceptación..."
          />
        </div>

        <button
          type="submit"
          disabled={isLoading}
          className="w-full py-3 px-6 rounded-lg text-white font-semibold bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-700 hover:to-indigo-700 shadow-md hover:shadow-lg transition duration-200 disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
        >
          {isLoading ? (
            <>
              <svg className="animate-spin h-5 w-5 text-white" fill="none" viewBox="0 0 24 24">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
              </svg>
              <span>Ejecutando inferencia neuronal (~23 ms)...</span>
            </>
          ) : (
            <span>Estimar Esfuerzo (Dual Decision Support)</span>
          )}
        </button>
      </form>

      {/* Manejo de Errores */}
      {error && (
        <div className="mt-6 p-4 bg-red-50 border-l-4 border-red-500 rounded-r-lg text-red-800 text-sm">
          <p className="font-semibold">Error al procesar la solicitud:</p>
          <p className="mt-1">{error}</p>
        </div>
      )}

      {/* Panel de Resultados: Soporte a la Decisión Dual */}
      {result && (
        <div className="mt-8 space-y-6 pt-6 border-t border-gray-200">
          <div className="flex items-center justify-between">
            <h2 className="text-xl font-bold text-gray-900">
              Evaluación Predictiva del Ticket <span className="font-mono text-blue-600">[{result.ticket_id}]</span>
            </h2>
            <span className="text-xs bg-gray-100 text-gray-600 px-2 py-1 rounded">
              Ecosistema: {selectedProject?.name || formData.project_domain}
            </span>
          </div>

          {/* Grid Dual: Red Neuronal vs Mediana Histórica */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
            {/* Tarjeta 1: Predicción Neuronal */}
            <div className="bg-gradient-to-br from-blue-50 to-indigo-50 border border-blue-200 rounded-xl p-5 shadow-sm relative overflow-hidden">
              <div className="absolute top-2 right-2 text-xs font-semibold uppercase tracking-wider text-blue-700 bg-blue-100 px-2 py-0.5 rounded">
                Neural AI
              </div>
              <h3 className="text-xs font-bold uppercase tracking-wider text-blue-800 mb-2">
                Predicción Secuencial (LSTM v2)
              </h3>
              
              <div className="flex items-baseline gap-2 mt-1">
                <span className="text-4xl md:text-5xl font-black text-blue-700">
                  {result.prediction.story_points_rounded}
                </span>
                <span className="text-xl font-bold text-blue-600">SP</span>
                <span className="text-xs text-gray-500 ml-1">
                  (Fibonacci ágil)
                </span>
              </div>

              <div className="mt-4 pt-3 border-t border-blue-100 space-y-1.5 text-xs text-gray-600">
                <div className="flex justify-between">
                  <span>Inferencia continua (log1p):</span>
                  <span className="font-mono font-semibold text-gray-800">
                    {result.prediction.story_points_raw} SP
                  </span>
                </div>
                <div className="flex justify-between">
                  <span>MAE global histórico (test):</span>
                  <span className="font-mono font-semibold text-indigo-700">
                    3.26 SP (cobertura 72.6%)
                  </span>
                </div>
                <div className="flex justify-between">
                  <span>RMSE temporal:</span>
                  <span className="font-mono font-semibold text-gray-700">
                    {result.prediction.rmse_temporal}
                  </span>
                </div>
              </div>
            </div>

            {/* Tarjeta 2: Baseline Mediana del Proyecto */}
            <div className="bg-gradient-to-br from-slate-50 to-gray-50 border border-gray-200 rounded-xl p-5 shadow-sm relative overflow-hidden">
              <div className="absolute top-2 right-2 text-xs font-semibold uppercase tracking-wider text-gray-700 bg-gray-200 px-2 py-0.5 rounded">
                Baseline
              </div>
              <h3 className="text-xs font-bold uppercase tracking-wider text-gray-800 mb-2">
                Mediana Histórica del Proyecto
              </h3>

              <div className="flex items-baseline gap-2 mt-1">
                <span className="text-4xl md:text-5xl font-black text-gray-800">
                  {result.prediction.project_median_baseline}
                </span>
                <span className="text-xl font-bold text-gray-600">SP</span>
                <span className="text-xs text-gray-500 ml-1">
                  (Referencia de dominio)
                </span>
              </div>

              <div className="mt-4 pt-3 border-t border-gray-200 space-y-1.5 text-xs text-gray-600">
                <div className="flex justify-between">
                  <span>Partición origen:</span>
                  <span className="font-medium text-gray-800">80% previo al test (train + val)</span>
                </div>
                <div className="flex justify-between">
                  <span>Prevención de fuga:</span>
                  <span className="font-semibold text-emerald-600">Preprocesamiento temporal</span>
                </div>
                <div className="flex justify-between">
                  <span>MAE global del baseline:</span>
                  <span className="font-mono font-semibold text-gray-700">2.98 SP</span>
                </div>
              </div>
            </div>
          </div>

          {/* Tarjeta de Soporte a la Decisión / Deliberación */}
          {renderDeliberationInsight()}

          {/* Metadatos y Rigor Científico */}
          <div className="bg-gray-50 rounded-lg p-4 text-xs text-gray-500 border border-gray-200">
            <p className="font-semibold text-gray-700 mb-1">Rigor Metodológico y Validez Empírica:</p>
            <p>
              El modelo neuronal evalúa la representación semántica de títulos y descripciones tras tokenización y padding a 100 términos con objetivo logarítmico. Sin embargo, conforme a la evidencia empírica del benchmark Deep-SE (23,313 issues, 16 proyectos), la heurística de mediana de proyecto mantiene paridad competitiva general. Se recomienda tratar esta herramienta como soporte a la deliberación humana en Planning Poker, no como sustituto autónomo del criterio del equipo de desarrollo.
            </p>
          </div>
        </div>
      )}
    </div>
  );
}