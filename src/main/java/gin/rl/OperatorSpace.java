package gin.rl;

import gin.edit.Edit;
import gin.edit.statement.*;
import gin.edit.matched.*;
import gin.edit.modifynode.*;
import gin.edit.llm.*;

import java.util.*;

/**
 * Defines the space of available mutation operators for RL-based selection.
 *
 * This class provides centralized access to all mutation operators that can
 * be used in genetic improvement, categorized into:
 * - Traditional operators: Classic GI mutations (delete, copy, replace, swap, move)
 * - LLM operators: Large Language Model-based code transformations
 *
 * For dissertation experiments, you can easily configure which operators
 * are available to the RL agent by using different getter methods.
 */
public class OperatorSpace {

    /**
     * Get all traditional statement-level operators.
     *
     * These are the classic genetic improvement operators that work by
     * copying, moving, or modifying statements within the AST.
     *
     * @return list of traditional Edit classes
     */
    public static List<Class<? extends Edit>> getStatementOperators() {
        return Arrays.asList(
            DeleteStatement.class,
            CopyStatement.class,
            ReplaceStatement.class,
            SwapStatement.class,
            MoveStatement.class
        );
    }

    /**
     * Get matched statement operators.
     *
     * These operators are type-aware and only replace statements with
     * statements of the same syntactic type (e.g., if with if, for with for).
     *
     * @return list of matched Edit classes
     */
    public static List<Class<? extends Edit>> getMatchedOperators() {
        return Arrays.asList(
            MatchedDeleteStatement.class,
            MatchedCopyStatement.class,
            MatchedReplaceStatement.class,
            MatchedSwapStatement.class
        );
    }

    /**
     * Get node modification operators.
     *
     * These operators modify individual nodes in the AST, such as
     * replacing binary operators (+, -, *, /) or unary operators (!, -).
     *
     * @return list of node modification Edit classes
     */
    public static List<Class<? extends Edit>> getModifyNodeOperators() {
        return Arrays.asList(
            BinaryOperatorReplacement.class,
            UnaryOperatorReplacement.class
        );
    }

    /**
     * Get all traditional (non-LLM) operators.
     *
     * This combines statement, matched, and node modification operators.
     *
     * @return list of all traditional Edit classes
     */
    public static List<Class<? extends Edit>> getTraditionalOperators() {
        List<Class<? extends Edit>> all = new ArrayList<>();
        all.addAll(getStatementOperators());
        all.addAll(getMatchedOperators());
        all.addAll(getModifyNodeOperators());
        return all;
    }

    /**
     * Get LLM-based operators.
     *
     * These operators use Large Language Models to generate code transformations:
     * - LLMMaskedStatement: Masks part of the code and asks LLM to fill in
     * - LLMReplaceStatement: Asks LLM to provide alternative implementations
     *
     * Note: LLM operators require API keys and are slower/more expensive than
     * traditional operators.
     *
     * @return list of LLM Edit classes
     */
    public static List<Class<? extends Edit>> getLLMOperators() {
        return Arrays.asList(
            LLMMaskedStatement.class,
            LLMReplaceStatement.class
        );
    }

    /**
     * Get all available operators (traditional + LLM).
     *
     * This is the full operator space for experiments comparing
     * traditional GI with LLM-augmented GI.
     *
     * @return list of all Edit classes
     */
    public static List<Class<? extends Edit>> getAllOperators() {
        List<Class<? extends Edit>> all = new ArrayList<>();
        all.addAll(getTraditionalOperators());
        all.addAll(getLLMOperators());
        return all;
    }

    /**
     * Get operators by category name.
     *
     * Useful for configuration-driven experiments.
     *
     * @param category one of: "statement", "matched", "modifynode", "traditional", "llm", "all"
     * @return list of Edit classes for that category
     * @throws IllegalArgumentException if category is unknown
     */
    public static List<Class<? extends Edit>> getOperatorsByCategory(String category) {
        return switch (category.toLowerCase()) {
            case "statement" -> getStatementOperators();
            case "matched" -> getMatchedOperators();
            case "modifynode", "modify_node" -> getModifyNodeOperators();
            case "traditional" -> getTraditionalOperators();
            case "llm" -> getLLMOperators();
            case "all" -> getAllOperators();
            default -> throw new IllegalArgumentException("Unknown operator category: " + category);
        };
    }

    /**
     * Check if an operator is LLM-based.
     *
     * @param operator the Edit class to check
     * @return true if operator is in the LLM package
     */
    public static boolean isLLMOperator(Class<? extends Edit> operator) {
        return operator.getPackage().getName().contains("llm");
    }

    /**
     * Get a human-readable name for an operator.
     *
     * @param operator the Edit class
     * @return short name (e.g., "DeleteStatement" instead of full class name)
     */
    public static String getOperatorName(Class<? extends Edit> operator) {
        return operator.getSimpleName();
    }

    /**
     * Get the category of an operator.
     *
     * @param operator the Edit class
     * @return category name: "llm", "statement", "matched", or "modifynode"
     */
    public static String getOperatorCategory(Class<? extends Edit> operator) {
        if (isLLMOperator(operator)) {
            return "llm";
        } else if (getStatementOperators().contains(operator)) {
            return "statement";
        } else if (getMatchedOperators().contains(operator)) {
            return "matched";
        } else if (getModifyNodeOperators().contains(operator)) {
            return "modifynode";
        } else {
            return "unknown";
        }
    }

    /**
     * Print summary of all available operators.
     */
    public static void printOperatorSummary() {
        System.out.println("=== Available Operators ===\n");

        System.out.println("Statement Operators (" + getStatementOperators().size() + "):");
        for (Class<? extends Edit> op : getStatementOperators()) {
            System.out.println("  - " + op.getSimpleName());
        }

        System.out.println("\nMatched Operators (" + getMatchedOperators().size() + "):");
        for (Class<? extends Edit> op : getMatchedOperators()) {
            System.out.println("  - " + op.getSimpleName());
        }

        System.out.println("\nNode Modification Operators (" + getModifyNodeOperators().size() + "):");
        for (Class<? extends Edit> op : getModifyNodeOperators()) {
            System.out.println("  - " + op.getSimpleName());
        }

        System.out.println("\nLLM Operators (" + getLLMOperators().size() + "):");
        for (Class<? extends Edit> op : getLLMOperators()) {
            System.out.println("  - " + op.getSimpleName());
        }

        System.out.println("\nTotal: " + getAllOperators().size() + " operators");
    }

    // Prevent instantiation
    private OperatorSpace() {}
}
