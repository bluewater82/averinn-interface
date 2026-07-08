import "./ResultsPanel.css";
import { formatSet } from "./formatSet";

/*****************************************************************************
 * ResultsPanel.jsx
 * 
 * Component used for debugging and proof of function for current stack.
 * 
 * Displays the results provided by the backend after verification run
 * has completed. Provides the results in a basic summary and table format.
 * 
 *****************************************************************************/


function ResultsPanel({ backendStatus }) {
    return (
        <div className="results-panel">
            <h2>Analysis Results</h2>

            {!backendStatus && (
                <p>Status: Waiting for verification run.</p>
            )}

            {backendStatus && (
                <div>
                    <p>Status: {backendStatus.success ? "Complete" : "Error"}</p>

                    {backendStatus.safety_result && (
                        <p>
                            <strong>{backendStatus.safety_result}</strong>
                        </p>
                    )}

                    <p>Return code: {backendStatus.returncode}</p>

                    {backendStatus.csv_summary && (
                        <div>
                            <h3>CSV Summary</h3>

                            <div className="set-summary-row">
                                <SetSummaryCard
                                    title="Initial Set"
                                    data={backendStatus.csv_summary.initial_set}
                                />

                                <SetSummaryCard
                                    title="Final Set"
                                    data={backendStatus.csv_summary.final_set}
                                />
                            </div>
                        </div>
                    )}

                    {backendStatus.stderr && (
                        <div>
                            <h3>Error Output</h3>
                            <pre>{backendStatus.stderr}</pre>
                        </div>
                    )}
                </div>
            )}
        </div>
    );
}

function SetSummaryCard({ title, data }) {
    return (
        <div className="set-summary-card">
            <h4>{title}</h4>

            <table className="set-table">
                <tbody>
                    {formatSet(data).map((variable) => (
                        <tr key={variable.name}>
                            <td>{variable.name}</td>
                            <td>[{variable.low}, {variable.high}]</td>
                        </tr>
                    ))}
                </tbody>
            </table>
        </div>
    );
}



export default ResultsPanel;