function ControlTypeCard({ currentNetType, setNetType }) {
    return (
        <div className="verification-card">
            <h2 className="verification-card-title">Network Type</h2>

            <div className="form-row-custom">
                <label>Dynamic vs Non-Dynamic</label>

                <select
                    className="form-select"
                    value={currentNetType}
                    onChange={(e) => setNetType(e.target.value)}
                >
                    <option value="Dynamic">Dynamic</option>
                    <option value="Non-Dynamic">Non-Dynamic</option>
                </select>
            </div>
        </div>
    );
}

export default ControlTypeCard;