"""Terraform infrastructure management tasks."""

from pathlib import Path

from invoke import task


@task
def init(c, environment="prod"):
    """Initialize Terraform for an environment.

    Args:
        environment: Environment to initialize (dev, staging, prod)
    """
    tf_dir = Path("infrastructure/terraform/environments") / environment
    if not tf_dir.exists():
        print(f"❌ Environment directory not found: {tf_dir}")
        return

    print(f"Initializing Terraform for {environment} environment...")
    with c.cd(str(tf_dir)):
        c.run("terraform init")


@task
def plan(c, environment="prod"):
    """Run terraform plan for an environment.

    Args:
        environment: Environment to plan (dev, staging, prod)
    """
    tf_dir = Path("infrastructure/terraform/environments") / environment
    if not tf_dir.exists():
        print(f"❌ Environment directory not found: {tf_dir}")
        return

    print(f"Running Terraform plan for {environment} environment...")
    with c.cd(str(tf_dir)):
        c.run("terraform plan")


@task
def apply(c, environment="prod", auto_approve=False):
    """Apply terraform changes for an environment.

    Args:
        environment: Environment to apply (dev, staging, prod)
        auto_approve: Skip interactive approval (use with caution)
    """
    tf_dir = Path("infrastructure/terraform/environments") / environment
    if not tf_dir.exists():
        print(f"❌ Environment directory not found: {tf_dir}")
        return

    auto_approve_flag = "-auto-approve" if auto_approve else ""
    print(f"Applying Terraform changes for {environment} environment...")

    with c.cd(str(tf_dir)):
        c.run(f"terraform apply {auto_approve_flag}")


@task
def destroy(c, environment="dev"):
    """Destroy terraform-managed infrastructure (dev/staging only).

    Args:
        environment: Environment to destroy (dev or staging only, not prod)
    """
    if environment == "prod":
        print("❌ Destroying prod environment is not allowed via this command")
        print("   For prod destruction, use terraform directly with appropriate safeguards")
        return

    tf_dir = Path("infrastructure/terraform/environments") / environment
    if not tf_dir.exists():
        print(f"❌ Environment directory not found: {tf_dir}")
        return

    print(f"⚠️  WARNING: This will destroy all resources in {environment} environment")
    response = input("Type 'yes' to confirm: ")
    if response.lower() != "yes":
        print("Aborted.")
        return

    with c.cd(str(tf_dir)):
        c.run("terraform destroy")


@task
def import_resource(c, resource, id, environment="prod"):
    """Import an existing GCP resource into Terraform state.

    Args:
        resource: Terraform resource address (e.g., module.storage.google_storage_bucket.dvc)
        id: GCP resource ID
        environment: Environment to import into
    """
    tf_dir = Path("infrastructure/terraform/environments") / environment
    if not tf_dir.exists():
        print(f"❌ Environment directory not found: {tf_dir}")
        return

    print(f"Importing resource: {resource}")
    print(f"GCP ID: {id}")

    with c.cd(str(tf_dir)):
        c.run(f'terraform import "{resource}" "{id}"')


@task
def import_all(c, environment="prod"):
    """Run the automated import script for existing resources.

    Args:
        environment: Environment to import into
    """
    tf_dir = Path("infrastructure/terraform/environments") / environment
    if not tf_dir.exists():
        print(f"❌ Environment directory not found: {tf_dir}")
        return

    script_path = Path("infrastructure/scripts/import-existing.sh")
    if not script_path.exists():
        print(f"❌ Import script not found: {script_path}")
        return

    print("Running automated import script...")
    print("This will import existing GCP resources into Terraform state")

    with c.cd(str(tf_dir)):
        c.run("bash ../../scripts/import-existing.sh")


@task
def validate(c, environment="prod"):
    """Validate Terraform configuration.

    Args:
        environment: Environment to validate
    """
    tf_dir = Path("infrastructure/terraform/environments") / environment
    if not tf_dir.exists():
        print(f"❌ Environment directory not found: {tf_dir}")
        return

    print(f"Validating Terraform configuration for {environment}...")
    with c.cd(str(tf_dir)):
        c.run("terraform validate")
        c.run("terraform fmt -check -recursive")


@task
def format(c):
    """Format all Terraform files."""
    tf_root = Path("infrastructure/terraform")
    if not tf_root.exists():
        print(f"❌ Terraform directory not found: {tf_root}")
        return

    print("Formatting Terraform files...")
    with c.cd(str(tf_root)):
        c.run("terraform fmt -recursive")


@task
def output(c, environment="prod", output_name=None):
    """Show Terraform outputs.

    Args:
        environment: Environment to show outputs for
        output_name: Specific output to show (optional)
    """
    tf_dir = Path("infrastructure/terraform/environments") / environment
    if not tf_dir.exists():
        print(f"❌ Environment directory not found: {tf_dir}")
        return

    with c.cd(str(tf_dir)):
        if output_name:
            c.run(f"terraform output {output_name}")
        else:
            c.run("terraform output")


@task
def state_list(c, environment="prod"):
    """List all resources in Terraform state.

    Args:
        environment: Environment to list resources for
    """
    tf_dir = Path("infrastructure/terraform/environments") / environment
    if not tf_dir.exists():
        print(f"❌ Environment directory not found: {tf_dir}")
        return

    with c.cd(str(tf_dir)):
        c.run("terraform state list")


@task
def state_show(c, resource, environment="prod"):
    """Show details of a specific resource in Terraform state.

    Args:
        resource: Resource address to show
        environment: Environment to query
    """
    tf_dir = Path("infrastructure/terraform/environments") / environment
    if not tf_dir.exists():
        print(f"❌ Environment directory not found: {tf_dir}")
        return

    with c.cd(str(tf_dir)):
        c.run(f'terraform state show "{resource}"')


@task
def docs(c):
    """Open Terraform documentation in browser."""
    print("Opening Terraform documentation...")
    print("- Terraform Setup: infrastructure/docs/TERRAFORM_SETUP.md")
    print("- Migration Guide: infrastructure/docs/MIGRATION.md")
    print("- Operations Runbook: infrastructure/docs/RUNBOOK.md")


@task
def check(c, environment="prod"):
    """Run all checks on Terraform configuration.

    Args:
        environment: Environment to check
    """
    print("Running Terraform configuration checks...")

    # Validate
    print("\n1. Validating configuration...")
    validate(c, environment)

    # Plan
    print("\n2. Running plan...")
    plan(c, environment)

    print("\n✅ All checks passed!")
