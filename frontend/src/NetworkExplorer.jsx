import { useEffect, useMemo } from "react";
import "./NetworkExplorer.css";

/**
 * Interactive first-pass visualization for an uploaded feed-forward network.
 *
 * The topology intentionally represents each layer as a compact card instead
 * of drawing every neuron and edge. Selecting a card reveals the layer's
 * activation, dimensions, and parameter counts in the inspector panel.
 */
function NetworkExplorer({
    isOpen,
    network,
    selectedLayerId,
    onSelectLayer,
    isLoading,
    error,
    onClose
}) {
    const selectedLayer = useMemo(
        () => network?.layers?.find(
            (layer) => layer.id === selectedLayerId
        ) ?? network?.layers?.[0] ?? null,
        [network, selectedLayerId]
    );

    useEffect(() => {
        if (!isOpen) return undefined;

        function handleKeyDown(event) {
            if (event.key === "Escape") onClose();
        }

        const previousOverflow = document.body.style.overflow;
        document.body.style.overflow = "hidden";
        window.addEventListener("keydown", handleKeyDown);

        return () => {
            document.body.style.overflow = previousOverflow;
            window.removeEventListener("keydown", handleKeyDown);
        };
    }, [isOpen, onClose]);

    if (!isOpen) return null;

    return (
        <div className="network-explorer-backdrop" onMouseDown={onClose}>
            <section
                className="network-explorer"
                role="dialog"
                aria-modal="true"
                aria-labelledby="network-explorer-title"
                onMouseDown={(event) => event.stopPropagation()}
            >
                <header className="network-explorer-header">
                    <div>
                        <p className="network-explorer-kicker">
                            Neural Network Explorer
                        </p>
                        <h2 id="network-explorer-title">
                            {network?.filename ?? "Network architecture"}
                        </h2>
                    </div>

                    <button
                        type="button"
                        className="network-explorer-close"
                        onClick={onClose}
                        aria-label="Close network explorer"
                    >
                        ×
                    </button>
                </header>

                {isLoading && (
                    <div className="network-explorer-status" role="status">
                        <span className="network-explorer-spinner" />
                        Inspecting the uploaded ONNX network…
                    </div>
                )}

                {!isLoading && error && (
                    <div className="network-explorer-error" role="alert">
                        <strong>Unable to open this network.</strong>
                        <span>{error}</span>
                    </div>
                )}

                {!isLoading && !error && network && (
                    <>
                        <div className="network-summary-strip">
                            <SummaryMetric
                                label="Network type"
                                value={capitalize(network.network_type)}
                            />
                            <SummaryMetric
                                label="Layers"
                                value={formatInteger(network.layer_count)}
                            />
                            <SummaryMetric
                                label="Parameters"
                                value={formatInteger(network.parameter_count)}
                            />
                            <SummaryMetric
                                label="Architecture"
                                value={network.layers
                                    .map((layer) => layer.size)
                                    .join(" → ")}
                            />
                        </div>

                        <div className="network-explorer-body">
                            <div className="network-topology-panel">
                                <div className="network-section-heading">
                                    <div>
                                        <p>Architecture</p>
                                        <h3>Layer topology</h3>
                                    </div>
                                    <span>Select a layer to inspect it</span>
                                </div>

                                <div className="network-topology-scroll">
                                    <div className="network-topology">
                                        {network.layers.map((layer, index) => (
                                            <div
                                                className="network-layer-step"
                                                key={layer.id}
                                            >
                                                <LayerCard
                                                    layer={layer}
                                                    isSelected={
                                                        selectedLayer?.id === layer.id
                                                    }
                                                    onSelect={() =>
                                                        onSelectLayer(layer.id)
                                                    }
                                                />

                                                {index < network.layers.length - 1 && (
                                                    <div
                                                        className="network-layer-connector"
                                                        aria-hidden="true"
                                                    >
                                                        <span />
                                                        <b>›</b>
                                                    </div>
                                                )}
                                            </div>
                                        ))}
                                    </div>
                                </div>
                            </div>

                            <LayerInspector layer={selectedLayer} />
                        </div>
                    </>
                )}
            </section>
        </div>
    );
}

function SummaryMetric({ label, value }) {
    return (
        <div className="network-summary-metric">
            <span>{label}</span>
            <strong>{value}</strong>
        </div>
    );
}

function LayerCard({ layer, isSelected, onSelect }) {
    const visibleNodes = Math.min(layer.size, 6);

    return (
        <button
            type="button"
            className={
                `network-layer-card network-layer-card--${layer.role} ` +
                `${isSelected ? "network-layer-card--selected" : ""}`
            }
            onClick={onSelect}
            aria-pressed={isSelected}
        >
            <span className="network-layer-role">{layer.role}</span>
            <strong>{layer.name}</strong>

            <div className="network-neuron-preview" aria-hidden="true">
                {Array.from({ length: visibleNodes }, (_, index) => (
                    <span key={index} />
                ))}
                {layer.size > visibleNodes && <b>+{layer.size - visibleNodes}</b>}
            </div>

            <span className="network-layer-size">
                {formatInteger(layer.size)} {layer.size === 1 ? "neuron" : "neurons"}
            </span>
            <span className="network-activation-badge">
                {layer.activation ?? "Input"}
            </span>
        </button>
    );
}

function LayerInspector({ layer }) {
    if (!layer) return null;

    return (
        <aside className="network-layer-inspector">
            <p className="network-inspector-kicker">Selected layer</p>
            <div className="network-inspector-title-row">
                <div>
                    <h3>{layer.name}</h3>
                    <span>{capitalize(layer.role)} layer</span>
                </div>
                <span className={`network-role-marker network-role-marker--${layer.role}`}>
                    {layer.id}
                </span>
            </div>

            <dl className="network-layer-facts">
                <Fact label="Neurons" value={formatInteger(layer.size)} />
                <Fact label="Activation" value={layer.activation ?? "None"} />
                <Fact
                    label="Parameters"
                    value={formatInteger(layer.parameter_count)}
                />
                <Fact
                    label="Weight matrix"
                    value={formatShape(layer.weight_shape)}
                />
                <Fact
                    label="Bias vector"
                    value={formatShape(layer.bias_shape)}
                />
            </dl>

            <div className="network-matrix-preview">
                <div>
                    <p>Weight inspection</p>
                    <strong>
                        {layer.weight_shape
                            ? `${formatShape(layer.weight_shape)} matrix`
                            : "No incoming weights"}
                    </strong>
                </div>
                <span>
                    Exact values and the heatmap will appear here in the next
                    visualization increment.
                </span>
            </div>
        </aside>
    );
}

function Fact({ label, value }) {
    return (
        <div>
            <dt>{label}</dt>
            <dd>{value}</dd>
        </div>
    );
}

function formatShape(shape) {
    return Array.isArray(shape) && shape.length > 0
        ? shape.join(" × ")
        : "—";
}

function formatInteger(value) {
    return Number.isFinite(Number(value))
        ? Number(value).toLocaleString()
        : "—";
}

function capitalize(value) {
    if (!value) return "—";
    return `${value.charAt(0).toUpperCase()}${value.slice(1)}`;
}

export default NetworkExplorer;
