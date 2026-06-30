interface Props {
  features: {
    length_score: number;
    entity_score: number;
    relation_score: number;
    hop_demand_score: number;
    semantic_score: number;
  };
  size?: number;
}

const LABELS = ['长度', '实体', '关系', '跳数', '语义'];
const KEYS: (keyof Props['features'])[] = ['length_score', 'entity_score', 'relation_score', 'hop_demand_score', 'semantic_score'];

export default function RadarChart({ features, size = 160 }: Props) {
  const cx = size / 2;
  const cy = size / 2;
  const radius = size * 0.35;
  const levels = 4;

  // Compute axis points
  const angleForIndex = (i: number) => -Math.PI / 2 + (i * 2 * Math.PI) / 5;

  const axisPoints = KEYS.map((_, i) => {
    const angle = angleForIndex(i);
    return {
      x: cx + radius * Math.cos(angle),
      y: cy + radius * Math.sin(angle),
    };
  });

  // Grid polygons for each level
  const gridPolygons = Array.from({ length: levels }, (_, level) => {
    const r = (radius * (level + 1)) / levels;
    return axisPoints
      .map((_, i) => {
        const angle = angleForIndex(i);
        return `${cx + r * Math.cos(angle)},${cy + r * Math.sin(angle)}`;
      })
      .join(' ');
  });

  // Data polygon
  const dataPoints = KEYS.map((key, i) => {
    const value = features[key] ?? 0;
    const angle = angleForIndex(i);
    return {
      x: cx + radius * value * Math.cos(angle),
      y: cy + radius * value * Math.sin(angle),
    };
  });
  const dataPolygon = dataPoints.map((p) => `${p.x},${p.y}`).join(' ');

  return (
    <svg width={size} height={size} className="block mx-auto">
      {/* Background grids */}
      {gridPolygons.map((points, idx) => (
        <polygon
          key={idx}
          points={points}
          fill="none"
          stroke="#e5e7eb"
          strokeWidth={1}
        />
      ))}

      {/* Axes */}
      {axisPoints.map((p, i) => (
        <line
          key={i}
          x1={cx}
          y1={cy}
          x2={p.x}
          y2={p.y}
          stroke="#e5e7eb"
          strokeWidth={1}
        />
      ))}

      {/* Data area */}
      <polygon
        points={dataPolygon}
        fill="rgba(99, 102, 241, 0.25)"
        stroke="#6366f1"
        strokeWidth={2}
      />

      {/* Data points */}
      {dataPoints.map((p, i) => (
        <circle key={i} cx={p.x} cy={p.y} r={3} fill="#6366f1" />
      ))}

      {/* Labels */}
      {axisPoints.map((_p, i) => {
        const angle = angleForIndex(i);
        const labelR = radius + 14;
        const lx = cx + labelR * Math.cos(angle);
        const ly = cy + labelR * Math.sin(angle);
        return (
          <text
            key={`label-${i}`}
            x={lx}
            y={ly}
            textAnchor="middle"
            dominantBaseline="middle"
            className="text-[10px] fill-gray-500"
          >
            {LABELS[i]}
          </text>
        );
      })}
    </svg>
  );
}
