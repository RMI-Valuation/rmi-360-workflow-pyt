# =============================================================================
# 🔐 Set AWS Keyring Credentials (tools/set_aws_keyring_tool.py)
# -----------------------------------------------------------------------------
# Tool Name:          SetAWSKeyringCredentialsTool
# Toolbox Context:    rmi_360_workflow.pyt
# Version:            1.1.0
# Author:             RMI Valuation, LLC
# Created:            2025-05-08
# Last Updated:       2025-05-15
#
# Description:
#   ArcPy Tool class for securely storing AWS credentials in the system keyring. These credentials
#   can be retrieved by other tools such as CopyToAwsTool to authenticate S3 or Lambda operations.
#   The credentials are stored under a configurable service name defined in the YAML config. Integrates
#   with Core Utils for configuration and secure credential handling.
#
# File Location:      /tools/set_aws_keyring_tool.py
# Core Utils:
#   - utils/manager/config_manager.py
#   - keyring
#
# Documentation:
#   See: docs_legacy/TOOL_GUIDES.md and docs_legacy/tools/copy_to_aws.md
#   (Ensure these docs are current; update if needed.)
#
# Parameters:
#   - AWS Access Key ID {access_key_id} (String): Your AWS access key. Will be stored securely in the keyring.
#   - AWS Secret Access Key {secret_access_key} (String): Your AWS secret key. Will be stored securely in the keyring.
#
# Notes:
#   - Stores both key and secret under the service name in config["copy_to_aws"]["keychain_service_name"].
#   - Raises runtime error if parameters are missing or storage fails.
#   - Ensure config file and keyring are properly configured for secure operations.
# =============================================================================

import arcpy
import keyring

from utils.manager.config_manager import ConfigManager


class SetAWSKeyringCredentialsTool(object):
    def __init__(self):
        self.label = "Set AWS Keyring Credentials"
        self.description = "Store AWS credentials securely in the system keyring for use by the Copy to AWS tool."
        self.category = "Setup"

    def getParameterInfo(self):
        return [
            arcpy.Parameter(
                displayName="AWS Access Key ID",
                name="access_key_id",
                datatype="GPString",
                parameterType="Required",
                direction="Input"
            ),
            arcpy.Parameter(
                displayName="AWS Secret Access Key",
                name="secret_access_key",
                datatype="GPString",
                parameterType="Required",
                direction="Input"
            )
        ]

    def execute(self, parameters, messages):
        """
        Stores AWS credentials in the system keyring for secure access by AWS-related tools.
        
        Extracts the AWS Access Key ID and Secret Access Key from input parameters and saves them
        to the system keyring under a configurable service name. Logs success or error messages
        using the provided messaging object.
        """
        try:
            cfg = ConfigManager.from_file(messages=messages)
            logger = cfg.get_logger()

            service_name = cfg.get("copy_to_aws.keychain_service_name", "rmi_s3")
            access_key_id = parameters[0].valueAsText
            secret_access_key = parameters[1].valueAsText

            if not (access_key_id and secret_access_key):
                logger.error("All parameters are required.", error_type=ValueError)

            keyring.set_password(service_name, "aws_access_key_id", access_key_id)
            keyring.set_password(service_name, "aws_secret_access_key", secret_access_key)

            logger.info(f"✅ AWS credentials saved to keyring under service '{service_name}'.")

        except Exception as e:
            logger.error(f"Failed to set AWS credentials: {e}", error_type=RuntimeError)
