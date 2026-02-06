/**
 * Monitoring Module
 * Comprehensive alerting and monitoring for CT Scan MLOps
 */

terraform {
  required_version = ">= 1.5.0"
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.0"
    }
  }
}

# Notification channels
resource "google_monitoring_notification_channel" "email_primary" {
  display_name = var.email_channel_display_name
  type         = "email"
  labels = {
    email_address = var.alert_email
  }
  enabled = true
}

resource "google_monitoring_notification_channel" "slack" {
  count        = var.slack_webhook_url != null ? 1 : 0
  display_name = "CT Scan Slack Channel"
  type         = "slack"
  labels = {
    channel_name = var.slack_channel_name
  }
  sensitive_labels {
    auth_token = var.slack_webhook_url
  }
  enabled = true
}

resource "google_monitoring_notification_channel" "pagerduty" {
  count        = var.pagerduty_key != null ? 1 : 0
  display_name = "CT Scan PagerDuty"
  type         = "pagerduty"
  sensitive_labels {
    service_key = var.pagerduty_key
  }
  enabled = true
}

# P0 Alert: API Downtime
resource "google_monitoring_alert_policy" "api_downtime" {
  display_name = "CT Scan API - Downtime (P0)"
  combiner     = "OR"
  enabled      = true

  conditions {
    display_name = "Health check fails for 1 minute"
    condition_threshold {
      filter          = "resource.type=\"cloud_run_revision\" AND resource.labels.service_name=\"${var.cloud_run_service_name}\" AND metric.type=\"run.googleapis.com/request_count\" AND metric.labels.response_code_class=\"5xx\""
      duration        = "60s"
      comparison      = "COMPARISON_GT"
      threshold_value = 0
      aggregations {
        alignment_period   = "60s"
        per_series_aligner = "ALIGN_RATE"
      }
    }
  }

  notification_channels = concat(
    [google_monitoring_notification_channel.email_primary.id],
    var.pagerduty_key != null ? [google_monitoring_notification_channel.pagerduty[0].id] : []
  )

  alert_strategy {
    auto_close = "1800s" # 30 minutes
  }

  documentation {
    content   = file("${path.module}/runbooks/api-downtime.md")
    mime_type = "text/markdown"
  }
}

# P0 Alert: High Error Rate
resource "google_monitoring_alert_policy" "high_error_rate" {
  display_name = "CT Scan API - High Error Rate (P0)"
  combiner     = "OR"
  enabled      = true

  conditions {
    display_name = "Error rate > 5% over 5 minutes"
    condition_threshold {
      filter          = "resource.type=\"cloud_run_revision\" AND resource.labels.service_name=\"${var.cloud_run_service_name}\" AND metric.type=\"run.googleapis.com/request_count\""
      duration        = "300s"
      comparison      = "COMPARISON_GT"
      threshold_value = var.high_error_rate_threshold
      aggregations {
        alignment_period     = "60s"
        per_series_aligner   = "ALIGN_RATE"
        cross_series_reducer = "REDUCE_SUM"
        group_by_fields      = ["metric.label.response_code_class"]
      }
      denominator_filter = "resource.type=\"cloud_run_revision\" AND resource.labels.service_name=\"${var.cloud_run_service_name}\" AND metric.type=\"run.googleapis.com/request_count\""
      denominator_aggregations {
        alignment_period     = "60s"
        per_series_aligner   = "ALIGN_RATE"
        cross_series_reducer = "REDUCE_SUM"
      }
    }
  }

  notification_channels = concat(
    [google_monitoring_notification_channel.email_primary.id],
    var.slack_webhook_url != null ? [google_monitoring_notification_channel.slack[0].id] : []
  )

  alert_strategy {
    auto_close = "3600s" # 1 hour
  }

  documentation {
    content   = file("${path.module}/runbooks/high-error-rate.md")
    mime_type = "text/markdown"
  }
}

# P0 Alert: Memory Exhaustion
resource "google_monitoring_alert_policy" "memory_exhaustion" {
  display_name = "CT Scan API - Memory Exhaustion (P0)"
  combiner     = "OR"
  enabled      = true

  conditions {
    display_name = "Memory utilization > 90% for 3 minutes"
    condition_threshold {
      filter          = "resource.type=\"cloud_run_revision\" AND resource.labels.service_name=\"${var.cloud_run_service_name}\" AND metric.type=\"run.googleapis.com/container/memory/utilizations\""
      duration        = "180s"
      comparison      = "COMPARISON_GT"
      threshold_value = var.memory_exhaustion_threshold
      aggregations {
        alignment_period     = "60s"
        per_series_aligner   = "ALIGN_PERCENTILE_99"
        cross_series_reducer = "REDUCE_MAX"
      }
    }
  }

  notification_channels = concat(
    [google_monitoring_notification_channel.email_primary.id],
    var.slack_webhook_url != null ? [google_monitoring_notification_channel.slack[0].id] : []
  )

  alert_strategy {
    auto_close = "1800s"
  }

  documentation {
    content   = file("${path.module}/runbooks/memory-exhaustion.md")
    mime_type = "text/markdown"
  }
}

# P0 Alert: Container Crash Loop
resource "google_monitoring_alert_policy" "crash_loop" {
  display_name = "CT Scan API - Container Crash Loop (P0)"
  combiner     = "OR"
  enabled      = true

  conditions {
    display_name = ">3 restarts in 10 minutes"
    condition_threshold {
      filter          = "resource.type=\"cloud_run_revision\" AND resource.labels.service_name=\"${var.cloud_run_service_name}\" AND metric.type=\"run.googleapis.com/container/billable_instance_time\""
      duration        = "0s"
      comparison      = "COMPARISON_GT"
      threshold_value = var.crash_loop_restart_threshold
      aggregations {
        alignment_period   = "600s"
        per_series_aligner = "ALIGN_DELTA"
      }
    }
  }

  notification_channels = concat(
    [google_monitoring_notification_channel.email_primary.id],
    var.pagerduty_key != null ? [google_monitoring_notification_channel.pagerduty[0].id] : []
  )

  alert_strategy {
    auto_close = "1800s"
  }

  documentation {
    content   = file("${path.module}/runbooks/crash-loop.md")
    mime_type = "text/markdown"
  }
}

# P1 Alert: High Latency
resource "google_monitoring_alert_policy" "high_latency" {
  display_name = "CT Scan API - High Latency (P1)"
  combiner     = "OR"
  enabled      = var.enable_p1_p2_alerts

  conditions {
    display_name = "P95 latency > 2 seconds for 5 minutes"
    condition_threshold {
      filter          = "resource.type=\"cloud_run_revision\" AND resource.labels.service_name=\"${var.cloud_run_service_name}\" AND metric.type=\"run.googleapis.com/request_latencies\""
      duration        = "300s"
      comparison      = "COMPARISON_GT"
      threshold_value = var.high_latency_threshold_ms
      aggregations {
        alignment_period     = "60s"
        per_series_aligner   = "ALIGN_DELTA"
        cross_series_reducer = "REDUCE_PERCENTILE_95"
      }
    }
  }

  notification_channels = concat(
    [google_monitoring_notification_channel.email_primary.id],
    var.slack_webhook_url != null ? [google_monitoring_notification_channel.slack[0].id] : []
  )

  alert_strategy {
    auto_close = "3600s"
  }

  documentation {
    content   = file("${path.module}/runbooks/high-latency.md")
    mime_type = "text/markdown"
  }
}

# P2 Alert: Elevated Error Rate
resource "google_monitoring_alert_policy" "elevated_error_rate" {
  display_name = "CT Scan API - Elevated Error Rate (P2)"
  combiner     = "OR"
  enabled      = var.enable_p1_p2_alerts

  conditions {
    display_name = "Error rate > 2% over 15 minutes"
    condition_threshold {
      filter          = "resource.type=\"cloud_run_revision\" AND resource.labels.service_name=\"${var.cloud_run_service_name}\" AND metric.type=\"run.googleapis.com/request_count\" AND metric.labels.response_code_class=\"5xx\""
      duration        = "900s"
      comparison      = "COMPARISON_GT"
      threshold_value = var.elevated_error_rate_threshold
      aggregations {
        alignment_period   = "60s"
        per_series_aligner = "ALIGN_RATE"
      }
    }
  }

  notification_channels = [google_monitoring_notification_channel.email_primary.id]

  alert_strategy {
    auto_close = "7200s" # 2 hours
  }

  documentation {
    content   = file("${path.module}/runbooks/elevated-error-rate.md")
    mime_type = "text/markdown"
  }
}

# P2 Alert: High CPU Usage
resource "google_monitoring_alert_policy" "high_cpu" {
  display_name = "CT Scan API - High CPU Usage (P2)"
  combiner     = "OR"
  enabled      = var.enable_p1_p2_alerts

  conditions {
    display_name = "CPU utilization > 80% for 10 minutes"
    condition_threshold {
      filter          = "resource.type=\"cloud_run_revision\" AND resource.labels.service_name=\"${var.cloud_run_service_name}\" AND metric.type=\"run.googleapis.com/container/cpu/utilizations\""
      duration        = "600s"
      comparison      = "COMPARISON_GT"
      threshold_value = var.high_cpu_threshold
      aggregations {
        alignment_period     = "60s"
        per_series_aligner   = "ALIGN_PERCENTILE_99"
        cross_series_reducer = "REDUCE_MAX"
      }
    }
  }

  notification_channels = [google_monitoring_notification_channel.email_primary.id]

  alert_strategy {
    auto_close = "3600s"
  }

  documentation {
    content   = "High CPU usage detected. Monitor for performance degradation."
    mime_type = "text/markdown"
  }
}
