package gin.rl;

import gin.edit.Edit;
import org.pmw.tinylog.Logger;

import java.io.Serial;
import java.util.*;

/**
 * Policy Gradient (REINFORCE-style) operator selector.
 *
 * This selector implements a softmax policy gradient method for the
 * multi-armed bandit problem. Instead of maintaining Q-values directly,
 * it maintains "preferences" for each operator and converts them to
 * selection probabilities using the softmax function:
 *
 *     π(a) = exp(H(a)) / Σ exp(H(b))
 *
 * where H(a) is the preference for operator a.
 *
 * After each selection, preferences are updated using a REINFORCE-style
 * gradient update:
 *
 *     H(a) += α * (R - R̄) * (1 - π(a))     if a was selected
 *     H(a) -= α * (R - R̄) * π(a)           otherwise
 *
 * where:
 * - α is the learning rate
 * - R is the reward received
 * - R̄ is the average reward (baseline)
 *
 * Advantages:
 * - Learns a stochastic policy directly
 * - Can adapt to non-stationary environments
 * - Soft preferences allow for nuanced exploration
 *
 * Disadvantages:
 * - More hyperparameters to tune (learning rate)
 * - Can be slower to converge than UCB
 * - Sensitive to reward scaling
 */
public class PolicyGradientSelector extends AbstractBanditSelector {

    @Serial
    private static final long serialVersionUID = 1L;

    /** Learning rate for preference updates */
    private final double alpha;

    /** Preference values for each operator (H values) */
    private final double[] preferences;

    /** Current policy (selection probabilities) */
    private double[] policy;

    /** Running average of rewards (baseline) */
    private double averageReward;

    /** Total rewards received (for computing average) */
    private double totalReward;

    /** Number of rewards received */
    private int rewardCount;

    // Additional logging for analysis
    private final List<double[]> preferencesLog;
    private final List<double[]> policyLog;
    private final List<Double> averageRewardLog;

    /**
     * Create a policy gradient selector.
     *
     * @param operators list of available Edit classes
     * @param alpha learning rate (typical values: 0.01 to 0.1)
     * @param rng random number generator
     * @throws IllegalArgumentException if alpha is not positive
     */
    public PolicyGradientSelector(List<Class<? extends Edit>> operators, double alpha, Random rng) {
        super(operators, rng);

        if (alpha <= 0) {
            throw new IllegalArgumentException("Learning rate alpha must be positive, got: " + alpha);
        }

        this.alpha = alpha;

        this.preferences = new double[operators.size()];
        Arrays.fill(preferences, 0.0);

        this.policy = computeSoftmax(preferences);

        this.averageReward = 0.0;
        this.totalReward = 0.0;
        this.rewardCount = 0;

        this.preferencesLog = new ArrayList<>();
        this.policyLog = new ArrayList<>();
        this.averageRewardLog = new ArrayList<>();

        preferencesLog.add(preferences.clone());
        policyLog.add(policy.clone());
        averageRewardLog.add(averageReward);

        Logger.info("Created PolicyGradientSelector with alpha=" + alpha);
    }

    /**
     * Compute softmax probabilities from preferences.
     *
     * π(a) = exp(H(a)) / Σ exp(H(b))
     *
     * Uses the log-sum-exp trick for numerical stability.
     *
     * @param prefs preference values
     * @return probability distribution
     */
    private double[] computeSoftmax(double[] prefs) {
        int n = prefs.length;
        double[] result = new double[n];

        double maxPref = Arrays.stream(prefs).max().orElse(0);

        double sum = 0;
        for (int i = 0; i < n; i++) {
            result[i] = Math.exp(prefs[i] - maxPref);
            sum += result[i];
        }

        for (int i = 0; i < n; i++) {
            result[i] /= sum;
        }

        return result;
    }

    @Override
    public Class<? extends Edit> select() {
        preSelect();

        double r = rng.nextDouble();
        double cumulative = 0;

        Class<? extends Edit> selected = null;
        for (int i = 0; i < operators.size(); i++) {
            cumulative += policy[i];
            if (r <= cumulative) {
                selected = operators.get(i);
                break;
            }
        }

        if (selected == null) {
            selected = operators.get(operators.size() - 1);
        }

        Logger.debug("PolicyGradient: selected " + selected.getSimpleName() +
                " (π=" + String.format("%.4f", policy[operators.indexOf(selected)]) + ")");

        postSelect(selected);
        return selected;
    }

    @Override
    public void updateQuality(Class<? extends Edit> operator, Long parentFitness,
                              Long childFitness, boolean success) {
        super.updateQuality(operator, parentFitness, childFitness, success);

        double reward = calculateReward(parentFitness, childFitness, success);
        int selectedIndex = operators.indexOf(operator);

        // REINFORCE gradient update
        for (int i = 0; i < operators.size(); i++) {
            if (i == selectedIndex) {
                // H(a) += α * (R - R̄) * (1 - π(a))
                preferences[i] += alpha * (reward - averageReward) * (1 - policy[i]);
            } else {
                // H(a) -= α * (R - R̄) * π(a)
                preferences[i] -= alpha * (reward - averageReward) * policy[i];
            }
        }

        policy = computeSoftmax(preferences);

        totalReward += reward;
        rewardCount++;
        averageReward = totalReward / rewardCount;

        preferencesLog.add(preferences.clone());
        policyLog.add(policy.clone());
        averageRewardLog.add(averageReward);

        Logger.debug("PolicyGradient: updated preferences, avg_reward=" +
                String.format("%.4f", averageReward));
    }

    public double[] getPolicy() {
        return policy.clone();
    }

    public Map<Class<? extends Edit>, Double> getPolicyMap() {
        Map<Class<? extends Edit>, Double> policyMap = new HashMap<>();
        for (int i = 0; i < operators.size(); i++) {
            policyMap.put(operators.get(i), policy[i]);
        }
        return policyMap;
    }

    public double[] getPreferences() {
        return preferences.clone();
    }

    public Map<Class<? extends Edit>, Double> getPreferencesMap() {
        Map<Class<? extends Edit>, Double> prefMap = new HashMap<>();
        for (int i = 0; i < operators.size(); i++) {
            prefMap.put(operators.get(i), preferences[i]);
        }
        return prefMap;
    }

    public double getAlpha() {
        return alpha;
    }

    public double getBaselineReward() {
        return averageReward;
    }

    public List<double[]> getPreferencesLog() {
        return Collections.unmodifiableList(preferencesLog);
    }

    public List<double[]> getPolicyLog() {
        return Collections.unmodifiableList(policyLog);
    }

    public List<Double> getAverageRewardLog() {
        return Collections.unmodifiableList(averageRewardLog);
    }

    @Override
    public void reset() {
        super.reset();

        Arrays.fill(preferences, 0.0);
        policy = computeSoftmax(preferences);

        averageReward = 0.0;
        totalReward = 0.0;
        rewardCount = 0;

        preferencesLog.clear();
        policyLog.clear();
        averageRewardLog.clear();

        preferencesLog.add(preferences.clone());
        policyLog.add(policy.clone());
        averageRewardLog.add(averageReward);
    }

    @Override
    public String toString() {
        return "PolicyGradientSelector{alpha=" + alpha +
                ", operators=" + operators.size() +
                ", avgReward=" + String.format("%.4f", averageReward) + "}";
    }
}
