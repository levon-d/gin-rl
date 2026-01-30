package gin.rl;

import gin.edit.Edit;
import org.pmw.tinylog.Logger;

import java.io.Serial;
import java.io.Serializable;
import java.util.*;

/**
 * Abstract base class for Multi-Armed Bandit (MAB) based operator selectors.
 *
 * This class provides the common infrastructure for bandit algorithms including:
 * - Tracking average quality estimates for each operator (arm)
 * - Counting how many times each operator has been selected
 * - Computing rewards based on fitness improvement
 * - Logging for analysis and debugging
 *
 * Subclasses implement specific selection strategies (epsilon-greedy, UCB, etc.)
 * by overriding the {@link #select()} method.
 *
 * The reward function uses a ratio-based approach:
 * - reward = parentFitness / childFitness
 * - reward > 1 means improvement (child is faster)
 * - reward = 1 means no change
 * - reward < 1 means degradation
 * - reward = 0 for failed mutations
 */
public abstract class AbstractBanditSelector implements OperatorSelector, Serializable {

    @Serial
    private static final long serialVersionUID = 1L;

    /** List of available operators (arms) */
    protected final List<Class<? extends Edit>> operators;

    /** Average quality estimate for each operator (Q-values) */
    protected final Map<Class<? extends Edit>, Double> averageQualities;

    /** Number of times each operator has been selected */
    protected final Map<Class<? extends Edit>, Integer> actionCounts;

    /** Number of successful applications for each operator */
    protected final Map<Class<? extends Edit>, Integer> successCounts;

    /** Number of failed applications for each operator */
    protected final Map<Class<? extends Edit>, Integer> failureCounts;

    /** Total reward accumulated for each operator */
    protected final Map<Class<? extends Edit>, Double> totalRewards;

    /** The most recently selected operator */
    protected Class<? extends Edit> previousOperator;

    /** Random number generator for stochastic selection */
    protected final Random rng;

    // ===== Logging for analysis =====

    /** Log of rewards received at each step */
    protected final List<Double> rewardLog;

    /** Log of quality estimates after each update */
    protected final List<Map<Class<? extends Edit>, Double>> qualityLog;

    /** Log of action counts after each update */
    protected final List<Map<Class<? extends Edit>, Integer>> actionCountLog;

    /** Log of which operator was selected at each step */
    protected final List<Class<? extends Edit>> selectionLog;

    /** Log of whether each step was successful */
    protected final List<Boolean> successLog;

    // ===== Sanity check counters =====

    /** Number of times select() has been called */
    protected int selectCallCount = 0;

    /** Number of times updateQuality() has been called */
    protected int updateCallCount = 0;

    /**
     * Create a new bandit selector with the given operators.
     *
     * @param operators list of Edit classes that can be selected
     * @param rng random number generator for stochastic decisions
     */
    public AbstractBanditSelector(List<Class<? extends Edit>> operators, Random rng) {
        if (operators == null || operators.isEmpty()) {
            throw new IllegalArgumentException("Operators list cannot be null or empty");
        }

        this.operators = new ArrayList<>(operators);
        this.rng = rng;

        this.averageQualities = new HashMap<>();
        this.actionCounts = new HashMap<>();
        this.successCounts = new HashMap<>();
        this.failureCounts = new HashMap<>();
        this.totalRewards = new HashMap<>();

        for (Class<? extends Edit> op : operators) {
            averageQualities.put(op, 0.0);
            actionCounts.put(op, 0);
            successCounts.put(op, 0);
            failureCounts.put(op, 0);
            totalRewards.put(op, 0.0);
        }

        this.rewardLog = new ArrayList<>();
        this.qualityLog = new ArrayList<>();
        this.actionCountLog = new ArrayList<>();
        this.selectionLog = new ArrayList<>();
        this.successLog = new ArrayList<>();

        this.qualityLog.add(new HashMap<>(averageQualities));
        this.actionCountLog.add(new HashMap<>(actionCounts));

        Logger.info("Initialized bandit selector with " + operators.size() + " operators");
        logOperatorSummary();
    }

    /**
     * Calculate the reward for a mutation based on fitness values.
     *
     * The reward is computed as the ratio of parent fitness to child fitness.
     * For execution time optimization:
     * - reward > 1 means the child is faster (improvement)
     * - reward = 1 means no change
     * - reward < 1 means the child is slower (degradation)
     * - reward = 0 for failed mutations (compilation error, test failure, etc.)
     *
     * @param parentFitness fitness before mutation (execution time in ns)
     * @param childFitness fitness after mutation (execution time in ns), or null if failed
     * @param success whether the mutation was successfully applied and tests passed
     * @return the computed reward value
     */
    protected double calculateReward(Long parentFitness, Long childFitness, boolean success) {
        if (!success || childFitness == null || childFitness <= 0) {
            return 0.0;
        }
        if (parentFitness == null || parentFitness <= 0) {
            Logger.warn("Invalid parent fitness: " + parentFitness);
            return 0.0;
        }
        return (double) parentFitness / childFitness;
    }

    @Override
    public void updateQuality(Class<? extends Edit> operator, Long parentFitness,
                              Long childFitness, boolean success) {
        // Sanity check: ensure select() was called before updateQuality()
        if (updateCallCount >= selectCallCount) {
            Logger.warn("updateQuality() called without matching select() call");
        }
        updateCallCount++;

        double reward = calculateReward(parentFitness, childFitness, success);

        int count = actionCounts.get(operator) + 1;
        actionCounts.put(operator, count);

        // Incremental mean: Q(a) = Q(a) + (r - Q(a)) / n(a)
        double oldQ = averageQualities.get(operator);
        double newQ = oldQ + (reward - oldQ) / count;
        averageQualities.put(operator, newQ);

        if (success) {
            successCounts.put(operator, successCounts.get(operator) + 1);
        } else {
            failureCounts.put(operator, failureCounts.get(operator) + 1);
        }

        totalRewards.put(operator, totalRewards.get(operator) + reward);

        rewardLog.add(reward);
        qualityLog.add(new HashMap<>(averageQualities));
        actionCountLog.add(new HashMap<>(actionCounts));
        successLog.add(success);

        Logger.debug(String.format("Updated %s: reward=%.4f, newQ=%.4f, count=%d, success=%b",
                operator.getSimpleName(), reward, newQ, count, success));
    }

    /**
     * Called by subclasses at the start of select() to perform common bookkeeping.
     *
     * This method:
     * - Increments the select call counter
     * - Performs sanity checks
     */
    protected void preSelect() {
        // Sanity check: updateQuality should have been called after previous select
        if (selectCallCount > 0 && updateCallCount < selectCallCount) {
            Logger.warn("select() called without updateQuality() for previous selection");
        }
        selectCallCount++;
    }

    /**
     * Called by subclasses at the end of select() to log the selection.
     *
     * @param selected the operator that was selected
     */
    protected void postSelect(Class<? extends Edit> selected) {
        previousOperator = selected;
        selectionLog.add(selected);
        Logger.debug("Selected operator: " + selected.getSimpleName() +
                     " (LLM: " + isLLMOperator(selected) + ")");
    }

    @Override
    public List<Class<? extends Edit>> getOperators() {
        return Collections.unmodifiableList(operators);
    }

    @Override
    public Class<? extends Edit> getPreviousOperator() {
        return previousOperator;
    }

    @Override
    public Map<Class<? extends Edit>, OperatorStats> getOperatorStatistics() {
        Map<Class<? extends Edit>, OperatorStats> stats = new HashMap<>();
        for (Class<? extends Edit> op : operators) {
            stats.put(op, new OperatorStats(
                actionCounts.get(op),
                averageQualities.get(op),
                successCounts.get(op),
                failureCounts.get(op),
                totalRewards.get(op)
            ));
        }
        return stats;
    }

    public List<Double> getRewardLog() {
        return Collections.unmodifiableList(rewardLog);
    }

    public List<Map<Class<? extends Edit>, Double>> getQualityLog() {
        return Collections.unmodifiableList(qualityLog);
    }

    public List<Map<Class<? extends Edit>, Integer>> getActionCountLog() {
        return Collections.unmodifiableList(actionCountLog);
    }

    public List<Class<? extends Edit>> getSelectionLog() {
        return Collections.unmodifiableList(selectionLog);
    }

    public List<Boolean> getSuccessLog() {
        return Collections.unmodifiableList(successLog);
    }

    public Map<Class<? extends Edit>, Double> getAverageQualities() {
        return Collections.unmodifiableMap(averageQualities);
    }

    public Map<Class<? extends Edit>, Integer> getActionCounts() {
        return Collections.unmodifiableMap(actionCounts);
    }

    public int getTotalSelections() {
        return actionCounts.values().stream().mapToInt(Integer::intValue).sum();
    }

    public Class<? extends Edit> getBestOperator() {
        return Collections.max(operators, Comparator.comparingDouble(averageQualities::get));
    }

    public double getCumulativeReward() {
        return rewardLog.stream().mapToDouble(Double::doubleValue).sum();
    }

    public double getAverageReward() {
        if (rewardLog.isEmpty()) {
            return 0.0;
        }
        return getCumulativeReward() / rewardLog.size();
    }

    /**
     * Log a summary of all operators and their current statistics.
     */
    public void logOperatorSummary() {
        Logger.info("=== Operator Summary ===");
        Logger.info(String.format("%-35s %8s %8s %10s %8s",
                "Operator", "Count", "AvgQ", "SuccRate", "LLM"));
        Logger.info("-".repeat(75));

        for (Class<? extends Edit> op : operators) {
            int count = actionCounts.get(op);
            double avgQ = averageQualities.get(op);
            int successes = successCounts.get(op);
            int failures = failureCounts.get(op);
            double successRate = (count > 0) ? (double) successes / count : 0.0;
            boolean isLLM = isLLMOperator(op);

            Logger.info(String.format("%-35s %8d %8.4f %9.2f%% %8s",
                    op.getSimpleName(), count, avgQ, successRate * 100, isLLM ? "Yes" : "No"));
        }
        Logger.info("=".repeat(75));
    }

    /**
     * Reset all statistics and logs.
     *
     * This is useful for running multiple experiments without creating new instances.
     */
    public void reset() {
        for (Class<? extends Edit> op : operators) {
            averageQualities.put(op, 0.0);
            actionCounts.put(op, 0);
            successCounts.put(op, 0);
            failureCounts.put(op, 0);
            totalRewards.put(op, 0.0);
        }

        previousOperator = null;
        selectCallCount = 0;
        updateCallCount = 0;

        rewardLog.clear();
        qualityLog.clear();
        actionCountLog.clear();
        selectionLog.clear();
        successLog.clear();

        qualityLog.add(new HashMap<>(averageQualities));
        actionCountLog.add(new HashMap<>(actionCounts));

        Logger.info("Bandit selector reset");
    }
}
