package gin.rl;

import gin.edit.Edit;
import org.pmw.tinylog.Logger;

import java.io.Serial;
import java.util.List;
import java.util.Random;

/**
 * Uniform random operator selector (baseline).
 *
 * This selector randomly chooses an operator with uniform probability,
 * ignoring any feedback about operator performance. It serves as a
 * baseline for comparing more sophisticated selection strategies.
 *
 * While this is not a true "learning" algorithm, it extends
 * AbstractBanditSelector to track statistics for comparison purposes.
 */
public class UniformSelector extends AbstractBanditSelector {

    @Serial
    private static final long serialVersionUID = 1L;

    /**
     * Create a uniform random selector.
     *
     * @param operators list of available Edit classes
     * @param rng random number generator
     */
    public UniformSelector(List<Class<? extends Edit>> operators, Random rng) {
        super(operators, rng);
        Logger.info("Created UniformSelector (random baseline)");
    }

    @Override
    public Class<? extends Edit> select() {
        preSelect();
        Class<? extends Edit> selected = operators.get(rng.nextInt(operators.size()));
        postSelect(selected);
        return selected;
    }

    @Override
    public String toString() {
        return "UniformSelector{operators=" + operators.size() + "}";
    }
}
