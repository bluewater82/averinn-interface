import { useEffect, useMemo, useState } from "react";
import "./ResultsPanel.css";
import SummaryCards from "./SummaryCards";
import IntervalOverview from "./IntervalOverview";
import VariableInspector from "./VariableInspector";

/*****************************************************************************
 * ResultsPanel.jsx
 *
 * Coordinates the Results dashboard and selected-variable state.
 *****************************************************************************/

function ResultsPanel({ backendStatus }) {
    const variables =
        backendStatus?.csv_summary?.variables ?? [];

    const [selectedVariableName, setSelectedVariableName] =
        useState("");

    useEffect(() => {
        if (variables.length === 0) {
            setSelectedVariableName("");
            return;
        }

        setSelectedVariableName((currentName) => {
            const variableStillExists =
                variables.some(
                    (variable) =>
                        variable.name === currentName
                );

            return variableStillExists
                ? currentName
                : variables[0].name;
        });
    }, [backendStatus, variables]);

    const selectedVariable = useMemo(
        () =>
            variables.find(
                (variable) =>
                    variable.name === selectedVariableName
            ) ?? null,
        [variables, selectedVariableName]
    );

    if (!backendStatus) {
        return (
            <div className="results-panel">
                <p>
                    Waiting for verification results.
                </p>
            </div>
        );
    }

    return (
        <div className="results-panel">
            <SummaryCards
                backendStatus={backendStatus}
            />

            {!backendStatus.success && (
                <VerificationError
                    backendStatus={backendStatus}
                />
            )}

            {backendStatus.success &&
                backendStatus.csv_summary && (
                    <>
                        <IntervalOverview
                            variables={variables}
                            selectedVariable={
                                selectedVariableName
                            }
                            onSelectVariable={
                                setSelectedVariableName
                            }
                        />

                        <VariableInspector
                            variable={selectedVariable}
                        />
                    </>
                )}
        </div>
    );
}


function VerificationError({ backendStatus }) {
    return (
        <section className="results-error-panel">
            <h2>Verification Failed</h2>

            <p>
                The AVERINN process exited with code{" "}
                <strong>
                    {backendStatus.returncode}
                </strong>.
            </p>

            {backendStatus.stderr && (
                <pre className="results-error-output">
                    {backendStatus.stderr}
                </pre>
            )}
        </section>
    );
}


export default ResultsPanel;