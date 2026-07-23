import { useEffect, useMemo, useState } from "react";
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
    networkFile,
    apiBaseUrl,
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

    const [selectedConnectionId, setSelectedConnectionId] = useState(null);
    const [connectionError, setConnectionError] = useState("");
    const [isLoadingConnection, setIsLoadingConnection] = useState(false);

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

    useEffect(() => {
        setSelectedConnectionId(null);
        setConnectionData(null);
        setConnectionError("");
        setIsLoadingConnection(false);
    }, [network]);

    if (!isOpen) return null;

    async function inspectConnection(destinationLayerId) {
        setSelectedConnectionId(destinationLayerId);
        setConnectionData(null);
        setConnectionError("");

        if (!networkFile) {
            setConnectionError(
                "The original uploaded ONNX file is no longer available."
            );
            return;
        }

        setIsLoadingConnection(true);

        const formData = new FormData();

        formData.append("network_file", networkFile);
        formData.append(
            "destination_layer_id",
            String(destinationLayerId)
        );

        try {
            const response = await fetch(
                `${apiBaseUrl}/visualize-network/connection`,
                {
                    method: "POST",
                    body: formData
                }
            );

            const responseData = await response.json();

            if (!response.ok) {
                throw new Error(
                    responseData.detail
                    || "Unable to inspect this connection."
                );
            }

            setConnectionData(responseData);
        } catch (requestError) {
            setConnectionError(
                requestError instanceof Error
                    ? requestError.message
                    : "Unable to inspect this connection."
            );
        } finally {
            setIsLoadingConnection(false);
        }
    }

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
                                                <ConnectionButton
                                                    sourceLayer={layer}
                                                    destinationLayer={network.layers[index + 1]}
                                                    isSelected={
                                                        selectedConnectionId
                                                        === network.layers[index + 1].id
                                                    }
                                                    onSelect={() =>
                                                        inspectConnection(
                                                            network.layers[index + 1].id
                                                        )
                                                    }
                                                />
                                            )}    
                                            </div>
                                        ))}
                                    </div>
                                </div>
                            </div>

                            <section
                                className="network-connection-preview"
                                aria-live="polite"
                            >
                                {!selectedConnectionId && (
                                    <p className="network-connection-prompt">
                                        Select an Inspect button between two layers to view
                                        that connection.
                                    </p>
                                )}

                                {isLoadingConnection && (
                                    <p className="network-connection-status">
                                        Loading connection data…
                                    </p>
                                )}

                                {connectionError && (
                                    <div
                                        className="network-connection-error"
                                        role="alert"
                                    >
                                        {connectionError}
                                    </div>
                                )}

                                {connectionData && !isLoadingConnection && (
                                    <ConnectionSummary connection={connectionData} />
                                )}
                            </section>

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

function ConnectionButton({
    sourceLayer,
    destinationLayer,
    isSelected,
    onSelect
}) {
    const shape = destinationLayer.weight_shape;

    return (
        <button
            type="button"
            className={
                "network-layer-connector " +
                (
                    isSelected
                        ? "network-layer-connector--selected"
                        : ""
                )
            }
            onClick={onSelect}
            aria-pressed={isSelected}
            aria-label={
                `Inspect weights from ${sourceLayer.name} ` +
                `to ${destinationLayer.name}`
            }
        >
            <span className="network-connector-shape">
                {formatShape(shape)}
            </span>

            <span
                className="network-connector-arrow"
                aria-hidden="true"
            >
                <i />
                <b>›</b>
            </span>

            <span className="network-connector-action">
                Inspect
            </span>
        </button>
    );
}

function ConnectionSummary({ connection }) {
    const [destinationCount, sourceCount] =
        connection.weight_shape;

    const weightCount =
        destinationCount * sourceCount;

    return (
        <div className="network-connection-summary">
            <div>
                <p className="network-connection-eyebrow">
                    Selected transformation
                </p>

                <h3>
                    {connection.source_layer_name}
                    {" → "}
                    {connection.destination_layer_name}
                </h3>

                <p>
                    <code>z = Wa + b</code>
                </p>
            </div>

            <dl className="network-connection-facts">
                <div>
                    <dt>Weight matrix</dt>
                    <dd>
                        {destinationCount}
                        {" × "}
                        {sourceCount}
                    </dd>
                </div>

                <div>
                    <dt>Weights</dt>
                    <dd>{weightCount.toLocaleString()}</dd>
                </div>

                <div>
                    <dt>Biases</dt>
                    <dd>
                        {connection.biases.length.toLocaleString()}
                    </dd>
                </div>

                <div>
                    <dt>Activation</dt>
                    <dd>{connection.activation || "Linear"}</dd>
                </div>
            </dl>
        </div>
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
