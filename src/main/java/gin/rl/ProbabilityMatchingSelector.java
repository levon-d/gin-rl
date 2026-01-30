package gin.rl;

import gin.edit.Edit;
import org.pmw.tinylog.Logger;

import java.io.Serial;
import java.util.*;

/**
 * Probability Matching operator selector.
 *
 * This selector implements a probability matching strategy where operators
 * are selected with probabilities proportional to their estimated quality.
 * Unlike epsilon-greedy which either exploits or explores uniformly, probability
 * matching provides a smoother exploration-exploitation balance.
 *
 * The probability of selecting operator a is:
 *
 *     p(a) = p_min + (1 - n * p_min) * Q(a) / Σ Q(b)
 *
 * where:
 * - p_min is the minimum probability for any operator (ensures exploration)
 * - n is the number of operators
 * - Q(a) is the average quality estimate for operator a
 *
 * When all Q-values are 0 (initially), probabilities are uniform.
 *
 * Advantages:
 * - Natural balance between exploration and exploitation
 * - Guaranteed minimum exploration of all operators
 * - Selection probability reflects confidence in operator quality
 *
 * Disadvantages:
 * - Sensitive to Q-value scaling
 * - May over-explore poor operators compared to UCB
 */
public class ProbabilityMatchingSelector extends AbstractBanditSelector {

    @Serial
    private static final long serialVersionUID = 1L;

    /** Minimum probability for each operator (ensures exploration) */
    private final double pMin;

    /** Current selection probabilities */
    private double[] probabilities;

    /** Log of probability distributions over time */
    private final List<double[]> probabilitiesLog;

    /**
     * Create a probability matching selector.
     *
     * @param operators list of available Edit classes
     * @param pMin minimum probability for each operator (typical: 0.01 to 0.1)
     * @param rng random number generator
     * @throws IllegalArgumentException if pMin is out of valid range
     */
    public ProbabilityMatchingSelector(List<Class<? extends Edit>> operators, double pMin, Random rng) {
        super(operators, rng);

        // Validate p_min: must be positive and n * p_min < 1
        if (pMin <= 0) {
            throw new IllegalArgumentException("pMin must be positive, got: " + pMin);
        }
        if (operators.size() * pMin >= 1.0) {
            throw new IllegalArgumentException(
                "pMin too large: " + operators.size() + " * " + pMin + " = " +
                (operators.size() * pMin) + " >= 1.0");
        }

        this.pMin = pMin;

        int n = operators.size();
        this.probabilities = new double[n];
        Arrays.fill(probabilities, 1.0 / n);

        this.probabilitiesLog = new ArrayList<>();
        probabilitiesLog.add(probabilities.clone());

        Logger.info("Created ProbabilityMatchingSelector with pMin=" + pMin);
    }

    /**
     * Update selection probabilities based on current Q-values.
     *
     * p(a) = p_min + (1 - n * p_min) * Q(a) / Σ Q(b)
     *
     * If all Q-values are 0, probabilities remain uniform.
     */
    private void updateProbabilities() {
        int n = operators.size();

        double totalQ = 0;
        for (Class<? extends Edit> op : operators) {
            totalQ += averageQualities.get(op);
        }

        if (totalQ <= 0) {
            Arrays.fill(probabilities, 1.0 / n);
        } else {
            double remainingProbability = 1.0 - n * pMin;
            for (int i = 0; i < n; i++) {
                Class<? extends Edit> op = operators.get(i);
                double q = averageQualities.get(op);
                probabilities[i] = pMin + remainingProbability * (q / totalQ);
            }
        }

        double sum = Arrays.stream(probabilities).sum();
        if (sum > 0) {
            for (int i = 0; i < n; i++) {
                probabilities[i] /= sum;
            }
        }
    }

    @Override
    public Class<? extends Edit> select() {
        preSelect();

        double r = rng.nextDouble();
        double cumulative = 0;

        Class<? extends Edit> selected = null;
        for (int i = 0; i < operators.size(); i++) {
            cumulative += probabilities[i];
            if (r <= cumulative) {
                selected = operators.get(i);
                break;
            }
        }

        if (selected == null) {
            selected = operators.get(operators.size() - 1);
        }

        Logger.debug("ProbabilityMatching: selected " + selected.getSimpleName() +
                " (p=" + String.format("%.4f", probabilities[operators.indexOf(selected)]) + ")");

        postSelect(selected);
        return selected;
    }

    @Override
    public void updateQuality(Class<? extends Edit> operator, Long parentFitness,
                              Long childFitness, boolean success) {
        super.updateQuality(operator, parentFitness, childFitness, success);
        updateProbabilities();
        probabilitiesLog.add(probabilities.clone());
        Logger.debug("ProbabilityMatching: updated probabilities");
    }

    public double[] getProbabilities() {
        return probabilities.clone();
    }

    public Map<Class<? extends Edit>, Double> getProbabilityMap() {
        Map<Class<? extends Edit>, Double> probMap = new HashMap<>();
        for (int i = 0; i < operators.size(); i++) {
            probMap.put(operators.get(i), probabilities[i]);
        }
        return probMap;
    }

    public double getPMin() {
        return pMin;
    }

    public List<double[]> getProbabilitiesLog() {
        return Collections.unmodifiableList(probabilitiesLog);
    }

    @Override
    public void reset() {
        super.reset();

        int n = operators.size();
        Arrays.fill(probabilities, 1.0 / n);

        probabilitiesLog.clear();
        probabilitiesLog.add(probabilities.clone());
    }

    @Override
    public String toString() {
        return "ProbabilityMatchingSelector{pMin=" + pMin +
                ", operators=" + operators.size() + "}";
    }
}
