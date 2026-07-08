export function formatSet(rawSet) {
    if (!rawSet) {
        return [];
    }

    const variables = [];

    for (let i = 0; i < 20; i++) {
        const lowKey = `${i}Low`;
        const highKey = `${i}High`;

        if (lowKey in rawSet && highKey in rawSet) {
            variables.push({
                name: `x${i}`,
                low: rawSet[lowKey],
                high: rawSet[highKey]
            });
        }
    }

    return variables;
}