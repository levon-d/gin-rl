package gin.rl;

import gin.edit.Edit;
import org.pmw.tinylog.Logger;

import java.io.Serial;
import java.util.*;

/**
 * Upper Confidence Bound (UCB) operator selector.
 *
 * This selector implements the UCB1 algorithm for the multi-armed bandit
 * problem. UCB balances exploration and exploitation by selecting the
 * operator that maximizes:
 *
 *     UCB(a) = Q(a) + c * sqrt(ln(t) / n(a))
 *
 * where:
 * - Q(a) is the average quality estimate for operator a
 * - c is the exploration constant (typically sqrt(2))
 * - t is the total number of selections made
 * - n(a) is the number of times operator a has been selected
 *
 * The second term is the "uncertainty bonus" - it increases for operators
 * that haven't been tried recently, encouraging exploration.
 *
 * Advantages:
 * - Principled exploration based on uncertainty
 * - Provably optimal regret bounds
 * - Automatically balances exploration/exploitation
 *
 * Disadvantages:
 * - Requires each arm to be selected at least once initially
 * - May over-explore in practice compared to epsilon-greedy
 */
public class UCBSelector extends AbstractBanditSelector {

    @Serial
    private static final long serialVersionUID = 1L;

    /** Exploration constant (controls exploration vs exploitation) */
    private final double c;

    /** Set of operators that haven't been selected yet (for initialization) */
    private final Set<Class<? extends Edit>> unselectedOperators;

    /**
     * Create a UCB selector.
     *
     * @param operators list of available Edit classes
     * @param c exploration constant (typically sqrt(2) ≈ 1.414)
     * @param rng random number generator (used for tie-breaking)
     * @throws IllegalArgumentException if c is negative
     */
    public UCBSelector(List<Class<? extends Edit>> operators, double c, Random rng) {
        super(operators, rng);

        if (c < 0) {
            throw new IllegalArgumentException("Exploration constant c must be non-negative, got: " + c);
        }

        this.c = c;
        this.unselectedOperators = new HashSet<>(operators);

        Logger.info("Created UCBSelector with c=" + c);
    }

    /**
     * Create a UCB selector with the default exploration constant sqrt(2).
     *
     * @param operators list of available Edit classes
     * @param rng random number generator
     */
    public UCBSelector(List<Class<? extends Edit>> operators, Random rng) {
        this(operators, Math.sqrt(2), rng);
    }

    @Override
    public Class<? extends Edit> select() {
        preSelect();

        Class<? extends Edit> selected;

        // Initialization phase: ensure each operator is selected at least once
        if (!unselectedOperators.isEmpty()) {
            List<Class<? extends Edit>> unselectedList = new ArrayList<>(unselectedOperators);
            selected = unselectedList.get(rng.nextInt(unselectedList.size()));
            unselectedOperators.remove(selected);

            Logger.debug("UCB: INITIALIZATION - selecting unselected operator (" +
                    unselectedOperators.size() + " remaining)");
        } else {
            int totalSelections = getTotalSelections();

            selected = Collections.max(operators, (a, b) -> {
                double ucbA = computeUCB(a, totalSelections);
                double ucbB = computeUCB(b, totalSelections);
                return Double.compare(ucbA, ucbB);
            });

            Logger.debug("UCB: selected " + selected.getSimpleName() +
                    " (UCB=" + String.format("%.4f", computeUCB(selected, totalSelections)) + ")");
        }

        postSelect(selected);
        return selected;
    }

    /**
     * Compute the UCB value for an operator.
     *
     * UCB(a) = Q(a) + c * sqrt(ln(t) / n(a))
     *
     * @param operator the operator to compute UCB for
     * @param totalSelections total number of selections made (t)
     * @return the UCB value
     */
    private double computeUCB(Class<? extends Edit> operator, int totalSelections) {
        double q = averageQualities.get(operator);
        int n = actionCounts.get(operator);

        if (n == 0) {
            return Double.MAX_VALUE;
        }

        double explorationBonus = c * Math.sqrt(Math.log(totalSelections) / n);
        return q + explorationBonus;
    }

    public Map<Class<? extends Edit>, Double> getUCBValues() {
        int totalSelections = getTotalSelections();
        Map<Class<? extends Edit>, Double> ucbValues = new HashMap<>();
        for (Class<? extends Edit> op : operators) {
            ucbValues.put(op, computeUCB(op, totalSelections));
        }
        return ucbValues;
    }

    public double getC() {
        return c;
    }

    public boolean isInitialized() {
        return unselectedOperators.isEmpty();
    }

    @Override
    public void reset() {
        super.reset();
        unselectedOperators.clear();
        unselectedOperators.addAll(operators);
    }

    @Override
    public String toString() {
        return "UCBSelector{c=" + c + ", operators=" + operators.size() +
                ", initialized=" + isInitialized() + "}";
    }
}
