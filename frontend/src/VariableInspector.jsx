import "./VariableInspector.css";

/*****************************************************************************
 * VariableInspector.jsx
 *
 * Displays detailed interval progression and derived metrics for one selected
 * state variable.
 *****************************************************************************/

function VariableInspector({ variable }) {
    if (!variable) {
        return null;
    }

    const firstInterval = variable.history[0];
    const finalInterval =
        variable.history[variable.history.length - 1];

    return (
        <section className="variable-inspector">
            <div className="variable-inspector-header">
                <div>
                    <p className="results-section-kicker">
                        Variable analysis
                    </p>

                    <h2>
                        Variable {variable.name}
                    </h2>

                    <p>
                        Detailed reachable-set progression for
                        the selected state variable.
                    </p>
                </div>

                <div className="variable-inspector-badge">
                    {variable.history.length} sets
                </div>
            </div>

            <div className="variable-inspector-metrics">
                <InspectorMetric
                    label="Initial Width"
                    value={formatNumber(
                        variable.initial_width
                    )}
                />

                <InspectorMetric
                    label="Final Width"
                    value={formatNumber(
                        variable.final_width
                    )}
                />

                <InspectorMetric
                    label="Width Change"
                    value={formatSignedNumber(
                        variable.width_change
                    )}
                    detail={formatPercentage(
                        variable.width_change_percent
                    )}
                />

                <InspectorMetric
                    label="Center Shift"
                    value={formatSignedNumber(
                        variable.center_shift
                    )}
                />
            </div>

            <div className="variable-history">
                {variable.history.map((interval, position) => (
                    <IntervalStage
                        key={interval.set_index}
                        interval={interval}
                        isFirst={position === 0}
                        isLast={
                            position ===
                            variable.history.length - 1
                        }
                    />
                ))}
            </div>

            <div className="variable-inspector-comparison">
                <div>
                    <span>Initial center</span>
                    <strong>
                        {formatNumber(firstInterval.center)}
                    </strong>
                </div>

                <span
                    className="variable-comparison-arrow"
                    aria-hidden="true"
                >
                    →
                </span>

                <div>
                    <span>Final center</span>
                    <strong>
                        {formatNumber(finalInterval.center)}
                    </strong>
                </div>
            </div>
        </section>
    );
}


function InspectorMetric({
    label,
    value,
    detail
}) {
    return (
        <div className="inspector-metric">
            <span className="inspector-metric-label">
                {label}
            </span>

            <strong className="inspector-metric-value">
                {value}
            </strong>

            {detail && (
                <span className="inspector-metric-detail">
                    {detail}
                </span>
            )}
        </div>
    );
}


function IntervalStage({
    interval,
    isFirst,
    isLast
}) {
    return (
        <article className="variable-stage">
            <div className="variable-stage-marker">
                <span
                    className={
                        `variable-stage-dot ${
                            isFirst
                                ? "variable-stage-dot--initial"
                                : isLast
                                    ? "variable-stage-dot--final"
                                    : ""
                        }`
                    }
                />

                {!isLast && (
                    <span className="variable-stage-line" />
                )}
            </div>

            <div className="variable-stage-content">
                <div className="variable-stage-heading">
                    <h3>{interval.label}</h3>

                    <span>
                        Set {interval.set_index}
                    </span>
                </div>

                <div className="variable-stage-values">
                    <StageValue
                        label="Lower Bound"
                        value={formatNumber(interval.low)}
                    />

                    <StageValue
                        label="Upper Bound"
                        value={formatNumber(interval.high)}
                    />

                    <StageValue
                        label="Width"
                        value={formatNumber(interval.width)}
                    />

                    <StageValue
                        label="Center"
                        value={formatNumber(interval.center)}
                    />
                </div>
            </div>
        </article>
    );
}


function StageValue({ label, value }) {
    return (
        <div className="variable-stage-value">
            <span>{label}</span>
            <strong>{value}</strong>
        </div>
    );
}


function formatNumber(value) {
    const numericValue = Number(value);

    if (!Number.isFinite(numericValue)) {
        return "—";
    }

    return numericValue.toFixed(3);
}


function formatSignedNumber(value) {
    const numericValue = Number(value);

    if (!Number.isFinite(numericValue)) {
        return "—";
    }

    const prefix =
        numericValue > 0 ? "+" : "";

    return `${prefix}${numericValue.toFixed(3)}`;
}


function formatPercentage(value) {
    const numericValue = Number(value);

    if (!Number.isFinite(numericValue)) {
        return "No percentage";
    }

    const prefix =
        numericValue > 0 ? "+" : "";

    return `${prefix}${numericValue.toFixed(2)}%`;
}


export default VariableInspector;