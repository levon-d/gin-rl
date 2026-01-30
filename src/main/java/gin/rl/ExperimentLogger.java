package gin.rl;

import gin.edit.Edit;
import org.pmw.tinylog.Logger;

import java.io.*;
import java.nio.file.*;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.*;

/**
 * Comprehensive logging for RL-GI experiments.
 *
 * This class provides structured logging of experimental data for analysis
 * in your dissertation. It exports data in CSV format for easy import into
 * Python/R/Excel for visualization and statistical analysis.
 *
 * Logs include:
 * - Per-step data: operator selected, success/failure, fitness, reward
 * - Per-operator statistics: selection count, success rate, average quality
 * - Configuration: algorithm parameters, operator space
 * - Summary: total reward, best fitness, improvement percentage
 */
public class ExperimentLogger implements Serializable {

    @Serial
    private static final long serialVersionUID = 1L;

    private final String experimentId;
    private final String outputDir;
    private final List<StepRecord> stepRecords;
    private final Map<String, String> configuration;
    private final long startTime;

    private long originalFitness;
    private long bestFitness;
    private String bestPatch;

    /**
     * Record for each step in the experiment.
     */
    public record StepRecord(
        int step,
        String operatorName,
        String operatorCategory,
        boolean isLLMOperator,
        boolean success,
        boolean isImprovement,
        Long parentFitness,
        Long childFitness,
        double reward,
        long stepDurationMs,      // Per-step operator invocation time
        long cumulativeTimeMs,    // Cumulative time since experiment start
        String patchDescription
    ) {}

    /**
     * Create a new experiment logger.
     *
     * @param experimentId unique identifier for this experiment
     * @param outputDir directory to write log files
     */
    public ExperimentLogger(String experimentId, String outputDir) {
        this.experimentId = experimentId;
        this.outputDir = outputDir;
        this.stepRecords = new ArrayList<>();
        this.configuration = new LinkedHashMap<>();
        this.startTime = System.currentTimeMillis();

        try {
            Files.createDirectories(Paths.get(outputDir));
        } catch (IOException e) {
            Logger.warn("Could not create output directory: " + outputDir);
        }

        Logger.info("ExperimentLogger initialized: " + experimentId);
    }

    /**
     * Create a logger with auto-generated experiment ID.
     *
     * @param outputDir directory to write log files
     */
    public ExperimentLogger(String outputDir) {
        this(generateExperimentId(), outputDir);
    }

    private static String generateExperimentId() {
        return "exp_" + LocalDateTime.now().format(DateTimeFormatter.ofPattern("yyyyMMdd_HHmmss"));
    }

    /**
     * Set configuration parameters for logging.
     */
    public void setConfiguration(String key, String value) {
        configuration.put(key, value);
    }

    /**
     * Set multiple configuration parameters.
     */
    public void setConfiguration(Map<String, String> config) {
        configuration.putAll(config);
    }

    /**
     * Set the original (baseline) fitness.
     */
    public void setOriginalFitness(long fitness) {
        this.originalFitness = fitness;
        this.bestFitness = fitness;
        configuration.put("original_fitness", String.valueOf(fitness));
    }

    /**
     * Log a single step of the experiment.
     *
     * @param step step number (1-indexed)
     * @param operator the operator class that was selected
     * @param success whether the mutation succeeded (compiled + passed tests)
     * @param parentFitness fitness before mutation
     * @param childFitness fitness after mutation (null if failed)
     * @param reward the reward value
     * @param stepDurationMs time taken for this operator invocation (ms)
     * @param patchDescription string description of the patch
     */
    public void logStep(int step, Class<? extends Edit> operator,
                        boolean success, Long parentFitness, Long childFitness,
                        double reward, long stepDurationMs, String patchDescription) {

        boolean isLLM = OperatorSpace.isLLMOperator(operator);
        String category = OperatorSpace.getOperatorCategory(operator);
        boolean isImprovement = success && childFitness != null && childFitness < parentFitness;

        StepRecord record = new StepRecord(
            step,
            operator.getSimpleName(),
            category,
            isLLM,
            success,
            isImprovement,
            parentFitness,
            childFitness,
            reward,
            stepDurationMs,
            System.currentTimeMillis() - startTime,
            patchDescription
        );

        stepRecords.add(record);

        // Track best fitness
        if (success && childFitness != null && childFitness < bestFitness) {
            bestFitness = childFitness;
            bestPatch = patchDescription;
        }

        Logger.debug(String.format("Step %d: %s %s reward=%.4f",
            step, operator.getSimpleName(), success ? "SUCCESS" : "FAILED", reward));
    }

    /**
     * Export step-by-step data to CSV.
     *
     * @param filename output filename (relative to outputDir)
     */
    public void exportStepsToCSV(String filename) throws IOException {
        Path path = Paths.get(outputDir, filename);
        try (PrintWriter writer = new PrintWriter(Files.newBufferedWriter(path))) {
            writer.println("step,operator,category,is_llm,success,is_improvement," +
                          "parent_fitness,child_fitness,reward,step_duration_ms,cumulative_time_ms,patch");

            for (StepRecord r : stepRecords) {
                writer.printf("%d,%s,%s,%b,%b,%b,%s,%s,%.6f,%d,%d,\"%s\"%n",
                    r.step,
                    r.operatorName,
                    r.operatorCategory,
                    r.isLLMOperator,
                    r.success,
                    r.isImprovement,
                    r.parentFitness != null ? r.parentFitness : "",
                    r.childFitness != null ? r.childFitness : "",
                    r.reward,
                    r.stepDurationMs,
                    r.cumulativeTimeMs,
                    escapeCsvString(r.patchDescription)
                );
            }
        }
        Logger.info("Exported steps to: " + path);
    }

    /**
     * Export per-operator statistics to CSV.
     *
     * @param selector the operator selector (to get learned Q-values)
     * @param filename output filename
     */
    public void exportOperatorStatsToCSV(OperatorSelector selector, String filename) throws IOException {
        Path path = Paths.get(outputDir, filename);

        Map<String, Integer> selectionCounts = new HashMap<>();
        Map<String, Integer> successCounts = new HashMap<>();
        Map<String, Double> totalRewards = new HashMap<>();
        Map<String, Integer> improvementCounts = new HashMap<>();

        for (StepRecord r : stepRecords) {
            selectionCounts.merge(r.operatorName, 1, Integer::sum);
            if (r.success) {
                successCounts.merge(r.operatorName, 1, Integer::sum);
            }
            totalRewards.merge(r.operatorName, r.reward, Double::sum);
            if (r.isImprovement) {
                improvementCounts.merge(r.operatorName, 1, Integer::sum);
            }
        }

        Map<Class<? extends Edit>, OperatorSelector.OperatorStats> stats = selector.getOperatorStatistics();

        try (PrintWriter writer = new PrintWriter(Files.newBufferedWriter(path))) {
            writer.println("operator,category,is_llm,selection_count,success_count,success_rate," +
                          "improvement_count,improvement_rate,total_reward,avg_reward,learned_q");

            for (Class<? extends Edit> op : selector.getOperators()) {
                String name = op.getSimpleName();
                String category = OperatorSpace.getOperatorCategory(op);
                boolean isLLM = OperatorSpace.isLLMOperator(op);

                int selections = selectionCounts.getOrDefault(name, 0);
                int successes = successCounts.getOrDefault(name, 0);
                int improvements = improvementCounts.getOrDefault(name, 0);
                double totalReward = totalRewards.getOrDefault(name, 0.0);

                double successRate = selections > 0 ? (double) successes / selections : 0;
                double improvementRate = selections > 0 ? (double) improvements / selections : 0;
                double avgReward = selections > 0 ? totalReward / selections : 0;

                OperatorSelector.OperatorStats opStats = stats.get(op);
                double learnedQ = opStats != null ? opStats.averageQuality() : 0;

                writer.printf("%s,%s,%b,%d,%d,%.6f,%d,%.6f,%.6f,%.6f,%.6f%n",
                    name, category, isLLM,
                    selections, successes, successRate,
                    improvements, improvementRate,
                    totalReward, avgReward, learnedQ
                );
            }
        }
        Logger.info("Exported operator stats to: " + path);
    }

    /**
     * Export experiment configuration to CSV.
     *
     * @param filename output filename
     */
    public void exportConfigToCSV(String filename) throws IOException {
        Path path = Paths.get(outputDir, filename);
        try (PrintWriter writer = new PrintWriter(Files.newBufferedWriter(path))) {
            writer.println("key,value");
            for (Map.Entry<String, String> entry : configuration.entrySet()) {
                writer.printf("%s,\"%s\"%n", entry.getKey(), escapeCsvString(entry.getValue()));
            }
        }
        Logger.info("Exported config to: " + path);
    }

    /**
     * Export summary statistics to CSV.
     *
     * @param filename output filename
     */
    public void exportSummaryToCSV(String filename) throws IOException {
        Path path = Paths.get(outputDir, filename);

        int totalSteps = stepRecords.size();
        int successfulSteps = (int) stepRecords.stream().filter(r -> r.success).count();
        int improvements = (int) stepRecords.stream().filter(r -> r.isImprovement).count();
        double totalReward = stepRecords.stream().mapToDouble(r -> r.reward).sum();

        int llmSelections = (int) stepRecords.stream().filter(r -> r.isLLMOperator).count();
        int llmSuccesses = (int) stepRecords.stream().filter(r -> r.isLLMOperator && r.success).count();
        int llmImprovements = (int) stepRecords.stream().filter(r -> r.isLLMOperator && r.isImprovement).count();

        int tradSelections = totalSteps - llmSelections;
        int tradSuccesses = successfulSteps - llmSuccesses;
        int tradImprovements = improvements - llmImprovements;

        double improvementPct = originalFitness > 0
            ? 100.0 * (originalFitness - bestFitness) / originalFitness
            : 0;

        try (PrintWriter writer = new PrintWriter(Files.newBufferedWriter(path))) {
            writer.println("metric,value");
            writer.printf("experiment_id,%s%n", experimentId);
            writer.printf("total_steps,%d%n", totalSteps);
            writer.printf("successful_steps,%d%n", successfulSteps);
            writer.printf("success_rate,%.6f%n", (double) successfulSteps / totalSteps);
            writer.printf("improvements,%d%n", improvements);
            writer.printf("improvement_rate,%.6f%n", (double) improvements / totalSteps);
            writer.printf("total_reward,%.6f%n", totalReward);
            writer.printf("avg_reward,%.6f%n", totalReward / totalSteps);
            writer.printf("original_fitness,%d%n", originalFitness);
            writer.printf("best_fitness,%d%n", bestFitness);
            writer.printf("improvement_pct,%.2f%n", improvementPct);
            writer.printf("llm_selections,%d%n", llmSelections);
            writer.printf("llm_successes,%d%n", llmSuccesses);
            writer.printf("llm_improvements,%d%n", llmImprovements);
            writer.printf("llm_success_rate,%.6f%n", llmSelections > 0 ? (double) llmSuccesses / llmSelections : 0);
            writer.printf("traditional_selections,%d%n", tradSelections);
            writer.printf("traditional_successes,%d%n", tradSuccesses);
            writer.printf("traditional_improvements,%d%n", tradImprovements);
            writer.printf("traditional_success_rate,%.6f%n", tradSelections > 0 ? (double) tradSuccesses / tradSelections : 0);
            writer.printf("runtime_ms,%d%n", System.currentTimeMillis() - startTime);
            writer.printf("best_patch,\"%s\"%n", escapeCsvString(bestPatch != null ? bestPatch : ""));
        }
        Logger.info("Exported summary to: " + path);
    }

    /**
     * Export all data to CSV files.
     *
     * Creates:
     * - {experimentId}_steps.csv
     * - {experimentId}_operators.csv
     * - {experimentId}_config.csv
     * - {experimentId}_summary.csv
     *
     * @param selector the operator selector
     */
    public void exportAll(OperatorSelector selector) throws IOException {
        exportStepsToCSV(experimentId + "_steps.csv");
        exportOperatorStatsToCSV(selector, experimentId + "_operators.csv");
        exportConfigToCSV(experimentId + "_config.csv");
        exportSummaryToCSV(experimentId + "_summary.csv");
        Logger.info("All data exported to: " + outputDir);
    }

    public String getExperimentId() {
        return experimentId;
    }

    public String getOutputDir() {
        return outputDir;
    }

    public List<StepRecord> getStepRecords() {
        return Collections.unmodifiableList(stepRecords);
    }

    public long getBestFitness() {
        return bestFitness;
    }

    public String getBestPatch() {
        return bestPatch;
    }

    /**
     * Print a summary to the console.
     */
    public void printSummary() {
        int totalSteps = stepRecords.size();
        int successfulSteps = (int) stepRecords.stream().filter(r -> r.success).count();
        int improvements = (int) stepRecords.stream().filter(r -> r.isImprovement).count();
        double totalReward = stepRecords.stream().mapToDouble(r -> r.reward).sum();

        double improvementPct = originalFitness > 0
            ? 100.0 * (originalFitness - bestFitness) / originalFitness
            : 0;

        System.out.println("\n" + "=".repeat(60));
        System.out.println("EXPERIMENT SUMMARY: " + experimentId);
        System.out.println("=".repeat(60));
        System.out.printf("Steps:            %d%n", totalSteps);
        System.out.printf("Successes:        %d (%.1f%%)%n", successfulSteps, 100.0 * successfulSteps / totalSteps);
        System.out.printf("Improvements:     %d (%.1f%%)%n", improvements, 100.0 * improvements / totalSteps);
        System.out.printf("Total Reward:     %.2f%n", totalReward);
        System.out.printf("Avg Reward:       %.4f%n", totalReward / totalSteps);
        System.out.printf("Original Fitness: %d ns%n", originalFitness);
        System.out.printf("Best Fitness:     %d ns%n", bestFitness);
        System.out.printf("Improvement:      %.2f%%%n", improvementPct);
        System.out.printf("Runtime:          %.1f s%n", (System.currentTimeMillis() - startTime) / 1000.0);
        System.out.println("=".repeat(60));
    }

    private String escapeCsvString(String s) {
        if (s == null) return "";
        return s.replace("\"", "\"\"").replace("\n", "\\n");
    }
}
