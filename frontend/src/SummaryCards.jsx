import "./SummaryCards.css";

/*****************************************************************************
 * SummaryCards.jsx
 *
 * Displays high-level statistics supplied by the backend results model.
 *****************************************************************************/

function SummaryCards({ backendStatus }) {
    const summary = backendStatus?.csv_summary;
    const statistics = summary?.statistics;

    const verdict = getVerdict(backendStatus);
    const verdictClass = getVerdictClass(verdict);

    return (
        <section
            className="summary-cards-section"
            aria-label="Verification summary"
        >
            <div className="summary-cards-grid">
                <SummaryCard
                    label="Verification Result"
                    value={verdict}
                    modifierClass={verdictClass}
                />

                <SummaryCard
                    label="State Variables"
                    value={summary?.variable_count ?? "—"}
                    detail={`${summary?.set_count ?? 0} reachable sets`}
                />

                <SummaryCard
                    label="Largest Final Width"
                    value={formatMetric(
                        statistics?.largest_final_width
                    )}
                    detail={
                        statistics?.widest_final_variable
                            ? `Variable ${statistics.widest_final_variable}`
                            : "Widest reachable interval"
                    }
                />

                <SummaryCard
                    label="Average Final Width"
                    value={formatMetric(
                        statistics?.average_final_width
                    )}
                    detail="Mean final interval width"
                />
            </div>
        </section>
    );
}


function SummaryCard({
    label,
    value,
    detail,
    modifierClass = ""
}) {
    return (
        <article className={`summary-card ${modifierClass}`}>
            <p className="summary-card-label">
                {label}
            </p>

            <p className="summary-card-value">
                {value}
            </p>

            {detail && (
                <p className="summary-card-detail">
                    {detail}
                </p>
            )}
        </article>
    );
}


function getVerdict(backendStatus) {
    if (!backendStatus) {
        return "Pending";
    }

    if (!backendStatus.success) {
        return "Error";
    }

    const safetyResult =
        backendStatus.safety_result?.trim();

    if (!safetyResult) {
        return "Complete";
    }

    const normalizedResult =
        safetyResult.toLowerCase();

    if (
        normalizedResult.includes("unsafe") ||
        normalizedResult.includes("violated")
    ) {
        return "Unsafe";
    }

    if (
        normalizedResult.includes("safe") ||
        normalizedResult.includes("verified")
    ) {
        return "Safe";
    }

    return safetyResult;
}


function getVerdictClass(verdict) {
    const normalizedVerdict =
        String(verdict).toLowerCase();

    if (normalizedVerdict === "safe") {
        return "summary-card--safe";
    }

    if (normalizedVerdict === "unsafe") {
        return "summary-card--unsafe";
    }

    if (normalizedVerdict === "error") {
        return "summary-card--error";
    }

    return "summary-card--neutral";
}


function formatMetric(value) {
    const numericValue = Number(value);

    if (!Number.isFinite(numericValue)) {
        return "—";
    }

    return numericValue.toFixed(3);
}


export default SummaryCards;