package gin.rl;

import gin.edit.Edit;
import org.pmw.tinylog.Logger;

import java.io.Serial;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;
import java.util.Random;

/**
 * Epsilon-Greedy operator selector.
 *
 * This selector implements the epsilon-greedy strategy for the multi-armed
 * bandit problem:
 * - With probability (1 - epsilon): exploit by selecting the operator with
 *   the highest average quality estimate
 * - With probability epsilon: explore by selecting a random operator
 *
 * Epsilon-greedy provides a simple but effective balance between exploration
 * and exploitation. Typical values for epsilon are 0.1 to 0.3.
 *
 * Advantages:
 * - Simple to implement and understand
 * - Constant exploration rate ensures continued learning
 * - Works well in stationary environments
 *
 * Disadvantages:
 * - Fixed exploration rate may be suboptimal
 * - Explores uniformly rather than focusing on uncertain options
 */
public class EpsilonGreedySelector extends AbstractBanditSelector {

    @Serial
    private static final long serialVersionUID = 1L;

    /** Exploration probability (0 to 1) */
    private final double epsilon;

    /**
     * Create an epsilon-greedy selector.
     *
     * @param operators list of available Edit classes
     * @param epsilon exploration probability (0 = pure exploitation, 1 = pure exploration)
     * @param rng random number generator
     * @throws IllegalArgumentException if epsilon is not in [0, 1]
     */
    public EpsilonGreedySelector(List<Class<? extends Edit>> operators, double epsilon, Random rng) {
        super(operators, rng);

        if (epsilon < 0 || epsilon > 1) {
            throw new IllegalArgumentException("Epsilon must be between 0 and 1, got: " + epsilon);
        }

        this.epsilon = epsilon;
        Logger.info("Created EpsilonGreedySelector with epsilon=" + epsilon);
    }

    @Override
    public Class<? extends Edit> select() {
        preSelect();

        Class<? extends Edit> selected;

        if (rng.nextDouble() < epsilon) {
            selected = operators.get(rng.nextInt(operators.size()));
            Logger.debug("Epsilon-greedy: EXPLORE - random selection");
        } else {
            selected = Collections.max(operators,
                    Comparator.comparingDouble(averageQualities::get));
            Logger.debug("Epsilon-greedy: EXPLOIT - best operator (Q=" +
                    String.format("%.4f", averageQualities.get(selected)) + ")");
        }

        postSelect(selected);
        return selected;
    }

    public double getEpsilon() {
        return epsilon;
    }

    @Override
    public String toString() {
        return "EpsilonGreedySelector{epsilon=" + epsilon + ", operators=" + operators.size() + "}";
    }
}
