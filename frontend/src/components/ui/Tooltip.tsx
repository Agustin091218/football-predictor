import * as TooltipPrimitive from "@radix-ui/react-tooltip";
import { type ReactNode } from "react";

const GLOSSARY: Record<string, string> = {
  "Poisson": "Distribución estadística que modela eventos raros e independientes. En fútbol: estima cuántos goles marca cada equipo por partido.",
  "Monte Carlo": "Simulación: corre el partido 10,000 veces con aleatoriedad para obtener probabilidades empíricas de cada marcador posible.",
  "XGBoost": "Algoritmo de machine learning que aprende patrones complejos a partir de 54 características del partido y 3,000 partidos históricos.",
  "Gemini": "Modelo de lenguaje de Google que analiza los datos del partido, busca en internet información reciente y genera explicaciones en lenguaje natural.",
  "log-loss": "Logarithmic Loss: mide qué tan buena es una predicción probabilística. 0 = predicción perfecta, 1.099 = aleatorio. Más bajo es mejor.",
  "Accuracy": "Porcentaje de predicciones correctas sobre el total. Un modelo aleatorio tendría ~33% (3 resultados posibles).",
  "BTTS": "Both Teams To Score: ambos equipos marcan al menos 1 gol en el partido.",
  "Over/Under": "Over 2.5: más de 2.5 goles totales (3 o más). Under 2.5: 2 goles o menos.",
  "1X2": "Los tres resultados posibles de un partido: Victoria Local (1), Empate (X), Victoria Visitante (2).",
  "Goles": "Precisión en la estimación de la diferencia de goles (xG diff vs goles reales). ±1 gol se considera acierto.",
  "Confianza": "Entropía de Shannon normalizada. Mide qué tan decidido está el modelo. Alta = un resultado domina claramente. Baja = partido muy parejo.",
  "Calibración Isotónica": "IsotonicRegression: ajusta las probabilidades para que '60% de chance' realmente signifique ~60% de aciertos en la práctica.",
  "Brier Score": "Error cuadrático medio de las probabilidades predichas. 0 = perfecto, 0.667 = aleatorio. Más bajo es mejor.",
  "Clean Sheet": "Portería a cero: el equipo NO recibe goles en todo el partido.",
};

export function Tooltip({ term, children }: { term: string; children: ReactNode }) {
  const explanation = GLOSSARY[term];
  if (!explanation) return <>{children}</>;

  return (
    <TooltipPrimitive.Provider delayDuration={200}>
      <TooltipPrimitive.Root>
        <TooltipPrimitive.Trigger asChild>
          <span className="cursor-help border-b border-dotted border-gray-500 hover:border-emerald-400 transition-colors">
            {children}
          </span>
        </TooltipPrimitive.Trigger>
        <TooltipPrimitive.Portal>
          <TooltipPrimitive.Content
            side="top"
            align="center"
            className="z-50 max-w-[320px] rounded-xl border border-[#252D3D] bg-[#0D0F14] px-4 py-3 text-sm text-gray-200 shadow-2xl animate-in fade-in-0 zoom-in-95"
            sideOffset={6}
          >
            <div className="font-semibold text-emerald-400 mb-1">{term}</div>
            <div className="text-xs text-gray-400 leading-relaxed">{explanation}</div>
            <TooltipPrimitive.Arrow className="fill-[#0D0F14]" />
          </TooltipPrimitive.Content>
        </TooltipPrimitive.Portal>
      </TooltipPrimitive.Root>
    </TooltipPrimitive.Provider>
  );
}
