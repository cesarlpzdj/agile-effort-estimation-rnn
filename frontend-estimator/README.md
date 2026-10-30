# Frontend Estimator: Dashboard de Soporte a la Decisión para Planning Poker

Este proyecto implementa la interfaz de usuario interactiva (Single Page Application - SPA) del sistema de soporte a la decisión para la estimación de esfuerzo ágil en Story Points.

Forma parte del trabajo de investigación:
> **Empirical Evaluation of Recurrent Neural Networks for Agile Effort Estimation**  
> *Revista Objetivo:* Empirical Software Engineering (Springer Nature)  
> *Autor:* César López de Jesús  
> *Afiliación:* Maestría en Inteligencia Artificial y Analítica de Datos (UACJ)

---

## 1. Filosofía de Diseño: Soporte a la Decisión Dual

En lugar de presentar el modelo de aprendizaje profundo como una caja negra infalible, la interfaz está concebida como un **sistema de soporte a la deliberación humana** para equipos Scrum y sesiones de Planning Poker:

1. **Indicador Semántico (Red Neuronal Recurrente):**
   Muestra la predicción continua (`story_points_raw`) y ajustada a la serie Fibonacci (`story_points_rounded`) calculada por la red `EMB_LSTM_v2` a partir del texto del ticket (título y descripción).
2. **Indicador de Calibración de Dominio (Mediana Histórica del Proyecto):**
   Muestra la mediana histórica específica del repositorio correspondiente, derivada del 80% de tickets cronológicamente anteriores al periodo de prueba (Clover 2.0, Jira Software 3.0, Moodle 8.0, etc.).
3. **Módulo de Análisis de Divergencia:**
   Cuando la predicción del modelo y la mediana del proyecto difieren por más de $2.0$ Story Points, la interfaz despliega una alerta visual neutral. Esto actúa como un catalizador de discusión técnica en el equipo (e.g., posible subestimación de tareas grandes o tareas atípicas para el estándar del proyecto), desalentando la aceptación acrítica de estimaciones automatizadas.
4. **Advertencia de Incertidumbre Poblacional:**
   Muestra el error medio absoluto histórico global ($3.26$ SP) acompañado explícitamente de su nota de cobertura empírica ($72.6\%$ en prueba cronológica), recordando a los estimadores que no constituye un intervalo de confianza paramétrico individual por ticket.

---

## 2. Tecnologías y Requisitos del Sistema

### Requisitos Previos
* **Node.js:** Versión 18.0.0 o superior (recomendado Node.js 20 LTS o 22 LTS).
* **Gestor de Paquetes:** `npm` (v9 o superior), `pnpm` o `yarn`.
* **Backend Activo:** El microservicio de inferencia [`effort-estimator`](../effort-estimator/) debe estar ejecutándose en `http://127.0.0.1:8000`.

### Pila Tecnológica
* **Framework:** React 19 (`react`, `react-dom` 19.2.x).
* **Empaquetador y Dev Server:** Vite 8 (`vite` 8.2.x).
* **Estilos y Diseño:** Tailwind CSS v4 (`tailwindcss`, `@tailwindcss/vite` 4.3.x).
* **Arquitectura:** Componentes funcionales modernos con React Hooks (`useState`).

---

## 3. Instalación y Puesta en Marcha

### Paso 1: Instalación de Dependencias
Desde la raíz del módulo `frontend-estimator`:
```bash
npm install
```

### Paso 2: Ejecución en Modo Desarrollo
Para iniciar el servidor de desarrollo local con recarga en caliente (HMR):
```bash
npm run dev
```
La aplicación estará disponible inmediatamente en:
```
http://localhost:5173
```

### Paso 3: Compilación para Producción
Para verificar o generar la distribución estática optimizada para despliegue:
```bash
npm run build
```
Los archivos optimizados se generarán en la carpeta `dist/`.

Para previsualizar localmente la versión de producción:
```bash
npm run preview
```

---

## 4. Estructura de Archivos del Frontend

```
frontend-estimator/
├── index.html                       # Documento HTML principal
├── package.json                     # Declaración de dependencias y scripts de npm
├── vite.config.js                   # Configuración del compilador Vite y Tailwind v4
├── eslint.config.js                 # Reglas de linting
├── public/                          # Activos estáticos públicos
└── src/
    ├── main.jsx                     # Punto de entrada de la aplicación React
    ├── App.jsx                      # Componente contenedor principal
    ├── index.css                    # Directivas de importación de estilos Tailwind CSS
    └── components/
        └── EffortEstimator.jsx      # Componente principal del estimador y panel de deliberación
```

---

## 5. Características del Dashboard

* **Selector de 16 Proyectos:** Catálogo completo de los 16 repositorios de Jira del benchmark Deep-SE con sus medianas históricas integradas:
  * Apache Mesos, Apache Usergrid, Appcelerator Studio, Aptana Studio, Atlassian Bamboo, Atlassian Clover, Atlassian Jira Software, Data Management, DuraCloud, Moodle LMS, Mule ESB, Mule Studio, Spring XD, Talend Data Quality, Talend ESB, Titanium Mobile SDK.
* **Cargador de Ejemplos Rápidos:** Botones de carga rápida con casos de uso reales de tickets complejos, historias y tareas (Moodle SSO, Mesos Allocator, DuraCloud S3 Sync, Clover Record Parser).
* **Selector de Tipos de Issue:** `Bug`, `Story`, `Task`, `Epic`, `Improvement`.
* **Manejo Resiliente de Errores:** Notificación automática en pantalla si el backend en el puerto 8000 no se encuentra disponible o si la red experimenta pérdidas de conexión.
* **Diseño Responsivo:** Interfaz adaptable a pantallas de escritorio, tablets y dispositivos móviles con paleta de colores limpia y accesible.

---

## 6. Comunicación con el Backend y Configuración de Red

Por defecto, el componente [`EffortEstimator.jsx`](src/components/EffortEstimator.jsx) realiza peticiones asíncronas vía `fetch` al endpoint:
```
POST http://127.0.0.1:8000/api/v1/predict
```

### Formato del Payload Enviado:
```json
{
  "ticket_id": "PRJ-101",
  "project_domain": "moodle",
  "title": "Descripción breve del problema",
  "description": "Texto detallado del requerimiento técnico",
  "issue_type": "Story"
}
```

Si necesitas cambiar la URL del backend (por ejemplo, si se despliega en un servidor remoto o contenedor Docker), modifica la constante de la petición en `src/components/EffortEstimator.jsx` o configura una variable de entorno `VITE_API_BASE_URL`.
