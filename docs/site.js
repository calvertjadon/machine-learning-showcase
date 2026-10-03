/* Machine learning studies static report.
   Reads measured scores from JSON files under results/.
   The page makes no external requests and invents no fallback scores.
   If a fetch fails, an alert explains the error. */

(function () {
  "use strict";

  var SOURCES = {
    nfl: "../results/metrics.json",
    linear: "../results/examples/linear_regression_metrics.json",
    classification: "../results/examples/classification_metrics.json",
    clustering: "../results/examples/clustering_metrics.json"
  };

  var MODEL_ORDER = ["random_forest", "logistic_regression", "down_distance", "majority"];

  var MODEL_LABELS = {
    random_forest: "Random forest (default)",
    logistic_regression: "Logistic regression",
    down_distance: "Down & distance baseline",
    majority: "Majority-class baseline"
  };

  var MODEL_SUMMARIES = {
    random_forest: "Highest accuracy of the four models; it trails logistic regression on macro F1 because it predicts few QB spikes.",
    logistic_regression: "Highest macro F1; it catches more QB spikes than the forest, at the cost of more false spike flags.",
    down_distance: "Rule-based reference built from training-set down-and-distance counts.",
    majority: "Predicts pass for every play; recall is zero for the other five classes."
  };

  var CLASS_LABELS = {
    field_goal: "Field goal",
    pass: "Pass",
    punt: "Punt",
    qb_kneel: "QB kneel",
    qb_spike: "QB spike",
    run: "Run"
  };

  var state = { nfl: null, linear: null, classification: null, clustering: null, failed: [] };

  function $(selector, root) {
    return (root || document).querySelector(selector);
  }

  function $$(selector, root) {
    return Array.prototype.slice.call((root || document).querySelectorAll(selector));
  }

  function setText(element, value) {
    if (element) {
      element.textContent = value;
    }
  }

  function el(tag, className, text) {
    var node = document.createElement(tag);
    if (className) {
      node.className = className;
    }
    if (text !== undefined) {
      node.textContent = text;
    }
    return node;
  }

  var intFormat = new Intl.NumberFormat("en-US");

  function pct(value, digits) {
    if (typeof value !== "number" || !isFinite(value)) {
      return "\u2013";
    }
    var places = digits === undefined ? 1 : digits;
    return (value * 100).toFixed(places) + "%";
  }

  function fixed(value, digits) {
    if (typeof value !== "number" || !isFinite(value)) {
      return "\u2013";
    }
    return value.toFixed(digits);
  }

  function scientific(value) {
    if (typeof value !== "number" || !isFinite(value)) {
      return "\u2013";
    }
    if (value === 0) {
      return "0";
    }
    return value.toExponential(1);
  }

  function model(key) {
    return state.nfl && state.nfl.models ? state.nfl.models[key] : null;
  }

  function classReport(m, name) {
    return m && m.per_class ? m.per_class[name] : null;
  }

  function confusion(m) {
    return m && m.confusion_matrix ? m.confusion_matrix : null;
  }

  function spikeStats(key) {
    var m = model(key);
    if (!m) {
      return null;
    }
    var report = classReport(m, "qb_spike");
    var matrix = confusion(m);
    var labels = m.labels || [];
    var index = labels.indexOf("qb_spike");
    var caught = null;
    var flagged = null;
    if (matrix && index >= 0) {
      caught = matrix[index] && typeof matrix[index][index] === "number" ? matrix[index][index] : null;
      flagged = 0;
      for (var r = 0; r < matrix.length; r += 1) {
        if (matrix[r] && typeof matrix[r][index] === "number") {
          flagged += matrix[r][index];
        }
      }
    }
    return {
      report: report,
      caught: caught,
      flagged: flagged,
      support: report && typeof report.support === "number" ? report.support : null,
      recall: report ? report.recall : null,
      f1: report ? report["f1-score"] : null
    };
  }

  function mix(counts) {
    if (!counts) {
      return null;
    }
    var entries = Object.keys(counts).map(function (name) {
      return { name: name, count: counts[name] };
    });
    entries.sort(function (a, b) {
      return b.count - a.count;
    });
    return entries
      .map(function (entry) {
        var label = CLASS_LABELS[entry.name] || entry.name;
        return label.toLowerCase() + " " + intFormat.format(entry.count);
      })
      .join(" \u00b7 ");
  }

  /* Values keyed by data-fill attributes in index.html. Each function reads
     the already-fetched JSON; returning null leaves the placeholder dash. */

  function fillValues() {
    var nfl = state.nfl;
    var meta = nfl && nfl.metadata ? nfl.metadata : null;
    var cleaning = meta && meta.source_cleaning ? meta.source_cleaning : null;
    var forest = model("random_forest");
    var logistic = model("logistic_regression");
    var spike = spikeStats("random_forest");
    var logisticSpike = spikeStats("logistic_regression");
    var fp = meta && meta.forest_params ? meta.forest_params : null;

    var values = {
      "nfl.cutoff": function () {
        return meta ? meta.holdout_start : null;
      },
      "nfl.train_rows": function () {
        return meta ? intFormat.format(meta.train_rows) : null;
      },
      "nfl.train_games": function () {
        return meta ? intFormat.format(meta.train_games) : null;
      },
      "nfl.train_dates": function () {
        return meta ? meta.train_date_min + " to " + meta.train_date_max : null;
      },
      "nfl.train_mix": function () {
        return meta ? mix(meta.train_label_counts) : null;
      },
      "nfl.test_rows": function () {
        return meta ? intFormat.format(meta.test_rows) : null;
      },
      "nfl.test_games": function () {
        return meta ? intFormat.format(meta.test_games) : null;
      },
      "nfl.test_dates": function () {
        return meta ? meta.test_date_min + " to " + meta.test_date_max : null;
      },
      "nfl.test_mix": function () {
        return meta ? mix(meta.test_label_counts) : null;
      },
      "nfl.seed": function () {
        return meta ? String(meta.seed) : null;
      },
      "nfl.python": function () {
        return meta ? meta.python_version : null;
      },
      "nfl.sklearn": function () {
        return meta ? meta.sklearn_version : null;
      },
      "nfl.pandas": function () {
        return meta ? "numpy " + meta.numpy_version + " \u00b7 pandas " + meta.pandas_version : null;
      },
      "nfl.settings": function () {
        if (!meta) {
          return null;
        }
        var depth = fp && typeof fp.max_depth === "number" ? fp.max_depth : null;
        var leaf = fp && typeof fp.min_samples_leaf === "number" ? fp.min_samples_leaf : null;
        return (
          "seed " + meta.seed + " \u00b7 " + meta.trees + " trees \u00b7 depth \u2264 " + depth +
          " \u00b7 min leaf " + leaf + " \u00b7 " + meta.jobs + " jobs"
        );
      },
      "nfl.source_file": function () {
        return meta ? meta.source_file : null;
      },
      "nfl.source_bytes": function () {
        if (!meta || typeof meta.source_bytes !== "number") {
          return null;
        }
        return intFormat.format(meta.source_bytes) + " bytes (\u2248 " + (meta.source_bytes / 1048576).toFixed(1) + " MiB)";
      },
      "nfl.sha256": function () {
        return meta ? meta.source_sha256 : null;
      },
      "nfl.cleaning.raw": function () {
        return cleaning ? intFormat.format(cleaning.raw_rows) : null;
      },
      "nfl.cleaning.dups": function () {
        return cleaning ? intFormat.format(cleaning.exact_duplicate_rows_removed) : null;
      },
      "nfl.cleaning.groups": function () {
        return cleaning ? intFormat.format(cleaning.ambiguous_id_groups) : null;
      },
      "nfl.cleaning.games": function () {
        return cleaning ? intFormat.format(cleaning.ambiguous_games_excluded) : null;
      },
      "nfl.cleaning.rows": function () {
        return cleaning ? intFormat.format(cleaning.ambiguous_rows_removed) : null;
      },
      "nfl.cleaning.retained": function () {
        return cleaning ? intFormat.format(cleaning.retained_rows) : null;
      },
      "nfl.forest.acc": function () {
        return forest ? pct(forest.accuracy) : null;
      },
      "nfl.forest.mf1": function () {
        return forest ? pct(forest.macro_f1) : null;
      },
      "nfl.logistic.mf1": function () {
        return logistic ? pct(logistic.macro_f1) : null;
      },
      "nfl.spike.caught": function () {
        return spike && spike.caught !== null ? String(spike.caught) : null;
      },
      "nfl.spike.support": function () {
        return spike && spike.support !== null ? String(spike.support) : null;
      },
      "nfl.spike.recall": function () {
        return spike ? pct(spike.recall) : null;
      },
      "nfl.spike.f1": function () {
        return spike ? pct(spike.f1) : null;
      },
      "nfl.spike.share": function () {
        if (!spike || spike.support === null || !meta || !meta.test_rows) {
          return null;
        }
        return pct(spike.support / meta.test_rows, 2);
      },
      "nfl.spike.logistic-caught": function () {
        return logisticSpike && logisticSpike.caught !== null ? String(logisticSpike.caught) : null;
      },
      "nfl.spike.logistic-recall": function () {
        return logisticSpike ? pct(logisticSpike.recall) : null;
      },
      "nfl.spike.logistic-f1": function () {
        return logisticSpike ? pct(logisticSpike.f1) : null;
      }
    };

    var linear = state.linear;
    if (linear) {
      var linData = linear.data || null;
      var linEstimators = linear.estimators || {};
      var linLstsq = linEstimators["analytical least squares"] || null;
      var linGd = linEstimators["batch gradient descent"] || null;
      var maxDiff = linear.agreement
        ? Math.max(
            linear.agreement.analytical_vs_gradient_descent_max_abs_coefficient_difference,
            linear.agreement.analytical_vs_sklearn_max_abs_coefficient_difference
          )
        : null;
      values["linear.n_features"] = function () {
        return linData ? String(linData.n_features) : null;
      };
      values["linear.r2"] = function () {
        return linLstsq ? fixed(linLstsq.test_r2, 4) : null;
      };
      values["linear.mse"] = function () {
        return linLstsq ? fixed(linLstsq.test_mse, 4) : null;
      };
      values["linear.agree"] = function () {
        return scientific(maxDiff);
      };
      values["linear.rows"] = function () {
        return linData ? linData.n_train + " train / " + linData.n_test + " held out" : null;
      };
      values["linear.true_coeffs"] = function () {
        if (!linData || !linData.true_coefficients) {
          return null;
        }
        return "[" + linData.true_coefficients.map(function (v) { return fixed(v, 2); }).join(", ") + "]";
      };
      values["linear.fitted_coeffs"] = function () {
        if (!linGd || !linGd.coefficients) {
          return null;
        }
        return "[" + linGd.coefficients.map(function (v) { return fixed(v, 2); }).join(", ") + "]";
      };
    }

    var classification = state.classification;
    if (classification) {
      var cancer = classification.breast_cancer || null;
      var knn = cancer ? cancer.knn : null;
      var gnb = cancer ? cancer.gaussian_nb : null;
      var toy = classification.toy_text || null;
      var toyCv = toy ? toy.cross_validation : null;
      var toyHoldout = toy ? toy.holdout : null;
      values["classification.samples"] = function () {
        return cancer ? String(cancer.n_samples) : null;
      };
      values["classification.features"] = function () {
        return cancer ? String(cancer.n_features) : null;
      };
      values["classification.knn.acc"] = function () {
        return knn ? pct(knn.accuracy) : null;
      };
      values["classification.knn.mf1"] = function () {
        return knn ? pct(knn.macro_f1) : null;
      };
      values["classification.gnb.acc"] = function () {
        return gnb ? pct(gnb.accuracy) : null;
      };
      values["classification.gnb.mf1"] = function () {
        return gnb ? pct(gnb.macro_f1) : null;
      };
      values["classification.holdout"] = function () {
        if (!cancer || !cancer.holdout) {
          return null;
        }
        return cancer.holdout.n_test + " held out (" + cancer.holdout.n_train + " train)";
      };
      values["classification.toy.n"] = function () {
        return toy ? toy.n_messages + " fictional messages" : null;
      };
      values["classification.toy.cv"] = function () {
        if (!toyCv) {
          return null;
        }
        return fixed(toyCv.macro_f1_mean, 2) + " \u00b1 " + fixed(toyCv.macro_f1_std, 2) + " (" + toyCv.folds + " folds)";
      };
      values["classification.toy.vocab"] = function () {
        if (!toy || !toy.token_indicators) {
          return null;
        }
        var vocab = toy.token_indicators.vocabulary_size;
        var trainVocab = toyHoldout ? toyHoldout.vocabulary_size_train : null;
        var trainText = typeof trainVocab === "number" ? intFormat.format(trainVocab) : "\u2013";
        return intFormat.format(vocab) + " terms. Training vocabulary has " + trainText + " terms.";
      };
    }

    var clustering = state.clustering;
    if (clustering) {
      var blobs = clustering.blobs || null;
      var selected = blobs ? blobs.selected : null;
      var image = clustering.image_quantization || null;
      var quantization = image ? image.quantization : null;
      var compression = image ? image.compression : null;
      values["clustering.samples"] = function () {
        return blobs ? intFormat.format(blobs.n_samples) : null;
      };
      values["clustering.k"] = function () {
        return blobs ? String(blobs.selected_k) : null;
      };
      values["clustering.silhouette"] = function () {
        return selected ? fixed(selected.silhouette, 3) : null;
      };
      values["clustering.ari"] = function () {
        return selected ? fixed(selected.adjusted_rand_index, 3) : null;
      };
      values["clustering.colors.source"] = function () {
        return quantization ? intFormat.format(quantization.unique_colors_source) : null;
      };
      values["clustering.colors.palette"] = function () {
        return quantization ? String(quantization.unique_colors_quantized) : null;
      };
      values["clustering.image.size"] = function () {
        return image ? image.width + " \u00d7 " + image.height + " (" + intFormat.format(image.n_pixels) + " pixels)" : null;
      };
      values["clustering.image.rmse"] = function () {
        return quantization ? fixed(quantization.rmse, 2) : null;
      };
      values["clustering.image.psnr"] = function () {
        return quantization ? fixed(quantization.psnr_db, 2) + " dB" : null;
      };
      values["clustering.image.ratio"] = function () {
        if (!compression) {
          return null;
        }
        return fixed(compression.index_stream_ratio, 1) + "\u00d7 index stream \u00b7 " + fixed(compression.total_ratio_with_palette, 2) + "\u00d7 with palette";
      };
    }

    return values;
  }

  function applyFills() {
    var values = fillValues();
    $$("[data-fill]").forEach(function (element) {
      var key = element.getAttribute("data-fill");
      var resolve = values[key];
      if (typeof resolve !== "function") {
        return;
      }
      var value = resolve();
      if (typeof value === "string" && value.length > 0) {
        element.textContent = value;
      }
    });
  }

  function renderScoreboard() {
    $$("[data-model-card]").forEach(function (card) {
      var key = card.getAttribute("data-model-card");
      var m = model(key);
      if (!m) {
        return;
      }
      var accuracy = card.querySelector('[data-v="acc"]');
      var macroF1 = card.querySelector('[data-v="mf1"]');
      var accuracyBar = card.querySelector('[data-v="acc-bar"]');
      var macroF1Bar = card.querySelector('[data-v="mf1-bar"]');
      setText(accuracy, pct(m.accuracy));
      setText(macroF1, pct(m.macro_f1));
      if (accuracy) {
        accuracy.title = "accuracy " + fixed(m.accuracy, 6);
      }
      if (macroF1) {
        macroF1.title = "macro F1 " + fixed(m.macro_f1, 6);
      }
      if (accuracyBar && typeof m.accuracy === "number") {
        accuracyBar.style.width = Math.max(0, Math.min(100, m.accuracy * 100)).toFixed(1) + "%";
      }
      if (macroF1Bar && typeof m.macro_f1 === "number") {
        macroF1Bar.style.width = Math.max(0, Math.min(100, m.macro_f1 * 100)).toFixed(1) + "%";
      }
    });
  }

  function renderSpikeTable() {
    var body = $("#spike-body");
    if (!body) {
      return;
    }
    body.textContent = "";
    if (!state.nfl) {
      var unavailable = el("tr");
      unavailable.appendChild(el("td", null, "Metrics unavailable. The file results/metrics.json could not be loaded; see the alert above."));
      unavailable.firstChild.colSpan = 5;
      body.appendChild(unavailable);
      return;
    }
    MODEL_ORDER.forEach(function (key) {
      var stats = spikeStats(key);
      if (!stats) {
        return;
      }
      var row = el("tr");
      if (key === "random_forest") {
        row.className = "is-highlight";
      }
      row.appendChild(el("th", null, MODEL_LABELS[key] || key)).scope = "row";
      row.appendChild(el("td", "num", stats.flagged === null ? "\u2013" : intFormat.format(stats.flagged)));
      row.appendChild(el("td", "num", stats.caught === null ? "\u2013" : intFormat.format(stats.caught) + " of " + intFormat.format(stats.support === null ? 0 : stats.support)));
      row.appendChild(el("td", "num", pct(stats.recall, 2)));
      row.appendChild(el("td", "num", pct(stats.f1, 2)));
      body.appendChild(row);
    });
  }

  function reportRow(label, report) {
    var row = el("tr");
    if (label === "QB spike") {
      row.className = "is-highlight";
    }
    row.appendChild(el("th", null, label)).scope = "row";
    ["precision", "recall", "f1-score", "support"].forEach(function (field) {
      var value = report ? report[field] : null;
      var cell = el("td", "num");
      if (field === "support") {
        cell.textContent = typeof value === "number" ? intFormat.format(value) : "\u2013";
      } else {
        cell.textContent = pct(value, 1);
        if (typeof value === "number") {
          cell.title = pct(value, 3);
        }
      }
      row.appendChild(cell);
    });
    return row;
  }

  function renderPerClass() {
    var select = $("#model-select");
    var body = $("#per-class-body");
    var foot = $("#per-class-foot");
    var caption = $("#per-class-caption");
    var summary = $("#per-class-summary");
    if (!body || !foot) {
      return;
    }
    var key = select ? select.value : "random_forest";
    var m = model(key);
    if (!m) {
      body.textContent = "";
      var row = el("tr");
      var cell = el("td", null, "Metrics unavailable. The file results/metrics.json could not be loaded; see the alert above.");
      cell.colSpan = 5;
      row.appendChild(cell);
      body.appendChild(row);
      setText(summary, "Unavailable. The file results/metrics.json could not be loaded.");
      return;
    }

    var label = MODEL_LABELS[key] || key;
    setText(caption, "Per-class metrics on the chronological holdout for " + label + ".");
    setText(
      summary,
      label + ": holdout accuracy " + pct(m.accuracy) + ", macro F1 " + pct(m.macro_f1) + ". " + (MODEL_SUMMARIES[key] || "")
    );

    body.textContent = "";
    var labels = state.nfl.metadata && state.nfl.metadata.target_labels ? state.nfl.metadata.target_labels : m.labels || [];
    labels.forEach(function (name) {
      body.appendChild(reportRow(CLASS_LABELS[name] || name, classReport(m, name)));
    });

    foot.textContent = "";
    foot.appendChild(reportRow("Macro average", classReport(m, "macro avg")));
    foot.appendChild(reportRow("Weighted average", classReport(m, "weighted avg")));
  }

  function markExampleUnavailable(exampleId) {
    var facts = $("#" + exampleId + " .facts--inline");
    if (facts) {
      var message = el("p", "field-note", "Example metrics unavailable. This JSON file could not be loaded; see the alert above. Figures shown below are static assets and may still be visible.");
      facts.parentNode.replaceChild(message, facts);
    }
  }

  function showFailures() {
    if (!state.failed.length) {
      return;
    }
    var alert = $("#data-alert");
    var detail = $("#data-alert-detail");
    if (!alert) {
      return;
    }
    var affected = state.failed.map(function (failure) {
      return failure.url + " (" + failure.message + ")";
    });
    setText(
      detail,
      "Failed to load: " + affected.join("; ") + ". Nothing on this page is invented to cover the gap; empty metrics stay blank."
    );
    alert.hidden = false;
  }

  function finish() {
    applyFills();
    if (state.nfl) {
      renderScoreboard();
    }
    renderSpikeTable();
    renderPerClass();

    ["linear", "classification", "clustering"].forEach(function (key) {
      var failed = state.failed.some(function (failure) {
        return failure.key === key;
      });
      if (!state[key] && failed) {
        markExampleUnavailable("example-" + key);
      }
    });

    showFailures();

    var status = $("#data-status");
    if (status) {
      status.hidden = true;
    }
    var main = $("#main");
    if (main) {
      main.setAttribute("aria-busy", "false");
    }
  }

  function loadAll() {
    var requests = Object.keys(SOURCES).map(function (key) {
      return fetch(SOURCES[key], { cache: "no-cache" })
        .then(function (response) {
          if (!response.ok) {
            throw new Error("HTTP " + response.status);
          }
          return response.json();
        })
        .then(function (json) {
          state[key] = json;
        })
        .catch(function (error) {
          state.failed.push({ key: key, url: SOURCES[key], message: error && error.message ? error.message : "fetch failed" });
        });
    });
    Promise.all(requests).then(finish, finish);
  }

  var select = $("#model-select");
  if (select) {
    select.addEventListener("change", renderPerClass);
  }

  loadAll();
})();
