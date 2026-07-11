import "./IntervalOverview.css";

/*****************************************************************************
 * IntervalOverview.jsx
 *
 * Displays all state-variable interval histories on a common scale.
 *****************************************************************************/

function IntervalOverview({
    variables,
    selectedVariable,
    onSelectVariable
}) {
    if (!Array.isArray(variables) || variables.length === 0) {
        return null;
    }

    const allBounds = variables.flatMap((variable) =>
        variable.history.flatMap((interval) => [
            interval.low,
            interval.high
        ])
    );

    const globalMinimum = Math.min(...allBounds);
    const globalMaximum = Math.max(...allBounds);
    const scaleRange =
        globalMaximum - globalMinimum || 1;

    const setLegend =
        variables[0]?.history ?? [];

    return (
        <section className="interval-overview">
            <div className="interval-overview-header">
                <div>
                    <p className="results-section-kicker">
                        Reachability visualization
                    </p>

                    <h2>Reachable Set Evolution</h2>

                    <p className="interval-overview-description">
                        Select a variable to inspect its initial,
                        intermediate, and final reachable bounds.
                    </p>
                </div>

                {/*<SetLegend intervals={setLegend} />*/}
            </div>

            {/*<div className="interval-scale-labels">
                <span>{formatNumber(globalMinimum)}</span>
                <span>{formatNumber(globalMaximum)}</span>
            </div>*/}

            <div className="interval-variable-list">
                {variables.map((variable) => (
                    <IntervalRow
                        key={variable.name}
                        variable={variable}
                        globalMinimum={globalMinimum}
                        scaleRange={scaleRange}
                        isSelected={
                            selectedVariable === variable.name
                        }
                        onSelect={() =>
                            onSelectVariable(variable.name)
                        }
                    />
                ))}
            </div>
        </section>
    );
}


function IntervalRow({
    variable,
    globalMinimum,
    scaleRange,
    isSelected,
    onSelect
}) {
    return (
        <button
            type="button"
            className={
                `interval-variable-row ${
                    isSelected
                        ? "interval-variable-row--selected"
                        : ""
                }`
            }
            onClick={onSelect}
            aria-pressed={isSelected}
        >
            <div className="interval-variable-name">
                {variable.name}
            </div>

            <div className="interval-row-content">
                <div
                    className="interval-track"
                    style={{
                        "--set-count": variable.history.length
                    }}
                >
                    {variable.history.map(
                        (interval, position) => (
                            <IntervalMark
                                key={interval.set_index}
                                interval={interval}
                                position={position}
                                totalSets={
                                    variable.history.length
                                }
                                globalMinimum={
                                    globalMinimum
                                }
                                scaleRange={scaleRange}
                            />
                        )
                    )}
                </div>

                <div className="interval-row-values">
                    {variable.history.map((interval) => (
                        <span key={interval.set_index}>
                            {interval.short_label}:{" "}
                            [{formatNumber(interval.low)},{" "}
                            {formatNumber(interval.high)}]
                        </span>
                    ))}
                </div>
            </div>
        </button>
    );
}


function IntervalMark({
    interval,
    position,
    totalSets,
    globalMinimum,
    scaleRange
}) {
    const intervalPosition = getIntervalPosition(
        interval.low,
        interval.high,
        globalMinimum,
        scaleRange
    );

    const verticalPosition =
        getVerticalPosition(position, totalSets);

    const setColor =
        getSetColor(position, totalSets);

    const sharedStyle = {
        left: `${intervalPosition.left}%`,
        top: `${verticalPosition}%`,
        "--set-color": setColor
    };

    const tooltip =
        `${interval.label}: ` +
        `[${formatNumber(interval.low)}, ` +
        `${formatNumber(interval.high)}]`;

    if (Math.abs(interval.width) < Number.EPSILON) {
        return (
            <div
                className="interval-point"
                style={sharedStyle}
                title={tooltip}
            />
        );
    }

    return (
        <div
            className="interval-range"
            style={{
                ...sharedStyle,
                width: `${intervalPosition.width}%`
            }}
            title={tooltip}
        />
    );
}


function SetLegend({ intervals }) {
    return (
        <div className="interval-overview-legend">
            {intervals.map((interval, position) => (
                <div
                    className="interval-legend-item"
                    key={interval.set_index}
                >
                    <span
                        className="interval-legend-swatch"
                        style={{
                            backgroundColor: getSetColor(
                                position,
                                intervals.length
                            )
                        }}
                    />

                    <span>{interval.short_label}</span>
                </div>
            ))}
        </div>
    );
}


function getIntervalPosition(
    low,
    high,
    globalMinimum,
    scaleRange
) {
    const left =
        ((low - globalMinimum) / scaleRange) * 100;

    const width =
        ((high - low) / scaleRange) * 100;

    return {
        left: clamp(left, 0, 100),
        width: clamp(width, 0.75, 100)
    };
}


function getVerticalPosition(position, totalSets) {
    if (totalSets <= 1) {
        return 50;
    }

    const top = 18;
    const bottom = 82;

    return (
        top +
        (position / (totalSets - 1)) *
            (bottom - top)
    );
}


function getSetColor(position, totalSets) {
    if (totalSets <= 1) {
        return "hsl(216 70% 45%)";
    }

    const progress =
        position / (totalSets - 1);

    const saturation =
        28 + progress * 52;

    const lightness =
        42 + progress * 10;

    return `hsl(
        216
        ${saturation}%
        ${lightness}%
    )`;
}


function clamp(value, minimum, maximum) {
    return Math.min(
        Math.max(value, minimum),
        maximum
    );
}


function formatNumber(value) {
    const numericValue = Number(value);

    if (!Number.isFinite(numericValue)) {
        return "—";
    }

    return Number(
        numericValue.toFixed(3)
    ).toString();
}


export default IntervalOverview;