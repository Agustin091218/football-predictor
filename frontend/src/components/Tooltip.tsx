import { type ReactNode, useState } from "react";

const EXPLANATIONS: Record<string, string> = {
  xG: "Expected Goals: goles esperados según el modelo. No es una predicción exacta, es un promedio estadístico.",
  BTTS: "Both Teams To Score: ambos equipos marcan al menos 1 gol.",
  "Over 2.5": "Más de 2.5 goles totales en el partido (3 o más goles).",
  "Clean Sheet": "El equipo NO recibe goles. Portería a cero.",
  Confianza: "Entropía de Shannon normalizada. Mide qué tan concentradas están las probabilidades. Alta = un resultado domina. Baja = partido parejo.",
  LogLoss: "Logarithmic Loss: mide qué tan buena es la predicción. Más bajo es mejor. 0 = perfecto, 1.1 = aleatorio.",
  "Brier Score": "Error cuadrático medio de las probabilidades. 0 = perfecto, 0.67 = aleatorio. Más bajo es mejor.",
  ELO: "Sistema de rating que mide la fuerza relativa de un equipo basado en resultados históricos (desde 1872).",
  "λ (lambda)": "Tasa de goles esperados en el modelo de Poisson. λ=2 significa que se esperan 2 goles en promedio.",
  "IC 95%": "Intervalo de Confianza 95%: rango donde cae el valor real con 95% de probabilidad.",
  "Upset Score": "Qué tan sorpresivo fue el resultado. 0% = esperado, 100% = totalmente inesperado.",
  "Calibración Isotónica": "Ajusta las probabilidades para que '60% de chance' realmente signifique ~60% de aciertos.",
  "Poisson": "Distribución estadística que modela eventos raros e independientes. En fútbol: goles por partido.",
  "Monte Carlo": "Simulación: corre el partido miles de veces con aleatoriedad para obtener probabilidades empíricas.",
};

export function Tooltip({ term, children }: { term: string; children: ReactNode }) {
  const [show, setShow] = useState(false);
  const explanation = EXPLANATIONS[term];

  if (!explanation) return <>{children}</>;

  return (
    <div
      className="relative inline-block cursor-help border-b border-dotted border-gray-500"
      onMouseEnter={() => setShow(true)}
      onMouseLeave={() => setShow(false)}
    >
      {children}
      {show && (
        <div className="absolute bottom-full left-1/2 -translate-x-1/2 mb-2 z-50 bg-[#0D0F14] border border-[#252D3D] rounded-lg p-3 text-xs text-gray-300 w-64 shadow-xl">
          <div className="font-semibold text-gray-200 mb-1">{term}</div>
          {explanation}
        </div>
      )}
    </div>
  );
}
