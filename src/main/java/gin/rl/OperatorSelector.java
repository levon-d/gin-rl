package gin.rl;

import gin.edit.Edit;

import java.util.List;
import java.util.Map;

/**
 * Interface for RL-based operator selection in Genetic Improvement.
 *
 * This interface defines the contract for selecting mutation operators
 * using reinforcement learning techniques. Implementations can use various
 * algorithms such as epsilon-greedy, UCB, or policy gradient methods.
 *
 * The RL agent treats each Edit class as an "arm" in a multi-armed bandit
 * problem, learning which operators are most effective for improving
 * program performance.
 */
public interface OperatorSelector {

    /**
     * Select an operator (edit class) to use for the next mutation.
     *
     * This method is called at each step of the local search to determine
     * which type of mutation to apply. The selection strategy depends on
     * the specific RL algorithm implementation.
     *
     * @return the selected Edit class to use for mutation
     */
    Class<? extends Edit> select();

    /**
     * Update the quality estimate for an operator based on the result of applying it.
     *
     * This method should be called after each mutation is evaluated to provide
     * feedback to the RL agent. The reward is typically calculated based on
     * the fitness improvement achieved.
     *
     * @param operator the operator (Edit class) that was used
     * @param parentFitness fitness (execution time in ns) before applying the edit
     * @param childFitness fitness after applying the edit, or null if the edit failed
     * @param success whether the edit was successfully applied and all tests passed
     */
    void updateQuality(Class<? extends Edit> operator, Long parentFitness,
                       Long childFitness, boolean success);

    /**
     * Get all operators that this selector can choose from.
     *
     * @return unmodifiable list of available Edit classes
     */
    List<Class<? extends Edit>> getOperators();

    /**
     * Get the most recently selected operator.
     *
     * Useful for logging and debugging purposes.
     *
     * @return the last operator selected by {@link #select()}, or null if none selected yet
     */
    Class<? extends Edit> getPreviousOperator();

    /**
     * Check if a given operator is an LLM-based operator.
     *
     * This is useful for analysis and logging to distinguish between
     * traditional GI operators and LLM-based operators.
     *
     * @param operator the Edit class to check
     * @return true if the operator is LLM-based, false otherwise
     */
    default boolean isLLMOperator(Class<? extends Edit> operator) {
        return operator.getPackage().getName().contains("llm");
    }

    /**
     * Get statistics about operator usage and performance.
     *
     * @return map of operator class to usage statistics
     */
    Map<Class<? extends Edit>, OperatorStats> getOperatorStatistics();

    /**
     * Record class for operator statistics.
     */
    record OperatorStats(
        int selectionCount,
        double averageQuality,
        int successCount,
        int failureCount,
        double totalReward
    ) {
        /**
         * Calculate the success rate for this operator.
         *
         * @return success rate as a value between 0 and 1, or 0 if never used
         */
        public double getSuccessRate() {
            int total = successCount + failureCount;
            return total > 0 ? (double) successCount / total : 0.0;
        }
    }
}
