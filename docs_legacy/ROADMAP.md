# 🛤️ RMI 360 Imaging Workflow Python Toolbox Roadmap

Welcome to the **RMI 360 Imaging Workflow Python Toolbox** development roadmap! This document outlines our strategic approach to creating a scalable, efficient, and innovative pipeline for processing and managing 360 imagery.

---

## 🔧 Short-Term Tasks  
**Goal**: These tasks require near-immediate attention and should be completed within the **v1.x** release line. They address critical requirements or improvements that are essential for initial project success.

### **1. 🌊 Watermark Support During Image Enhancement**  
![Effort](https://img.shields.io/badge/effort-MEDIUM-orange) ![Category](https://img.shields.io/badge/category-Refinement-orange)
**Description**: Add the capability to overlay **configurable text or logos** on processed images.  
- **Action Items**:
  - Support watermark position, opacity, and size customization.
  - Integrate watermark functionality during image enhancement. 
- **Development Plan**: [View Development Plan](../dev_docs/short_term/watermark_support.md)
📅 **Timeline**: Week 1-2: Implementation of watermark functionality.

### **2. 🔑 Lambda IAM Role Validation**  
![Effort](https://img.shields.io/badge/effort-LOW-green) ![Category](https://img.shields.io/badge/category-Foundational-blue)
**Description**: Implement checks to verify that **proper IAM permissions** are set up before deploying Lambda functions.  
- **Action Items**:
  - Create Lambda IAM role validation function to ensure proper permissions.
  - Add error handling for permission issues during deployment.
- **Development Plan**: [View Development Plan](../dev_docs/short_term/lambda_iam_validation.md)
📅 **Timeline**: Week 2: IAM validation setup.

### **3. 💻 Optional CLI Runner for Pipelines**  
![Effort](https://img.shields.io/badge/effort-MEDIUM-orange) ![Category](https://img.shields.io/badge/category-Flexibility-yellow)
**Description**: Provide an **optional CLI runner** for pipelines, allowing users to execute processing workflows independently of ArcGIS Pro.  
- **Action Items**:
  - Develop the CLI runner that triggers the image processing pipeline.
  - Provide CLI arguments for configuration, input/output directories, and other options.
- **Development Plan**: [View Development Plan](../dev_docs/short_term/cli_runner.md)
📅 **Timeline**: Week 3: CLI development and testing.

---

## **Phase 1: Foundational & Immediate Priorities**  
**Goal**: Establish a solid foundation for the Toolbox, focusing on efficient metadata management, flexible config handling, and a reprocessable workflow.

### **1. 📍 Centralizing Metadata Management and Gathering**  
![Effort](https://img.shields.io/badge/effort-HIGH-red) ![Category](https://img.shields.io/badge/category-Foundational-blue)  
**Description**: Create a centralized **SQLite database** to store all metadata, enabling easy updates and non-destructive edits.  
- **Action Items**:
  - Implement metadata capture during image collection.
  - Integrate metadata into the processing pipeline for easy access.  
- **Development Plan**: [View Development Plan](../dev_docs/phase_1/centralizing_metadata.md)  
- **Expected Outcome**: Streamlined metadata management, facilitating easy reprocessing.  
📅 **Timeline**: Week 1-2: Database setup, Week 3-4: Integration with pipeline.

### **2. 🔧 Refining Config Handling and Runtime Management**  
![Effort](https://img.shields.io/badge/effort-MEDIUM-orange) ![Category](https://img.shields.io/badge/category-Foundational-blue)  
**Description**: Introduce **RuntimeManager** to centralize config handling, ensuring dynamic access to configurations.  
- **Action Items**:
  - Refactor existing tools to fetch config from the centralized runtime database.  
- **Development Plan**: [View Development Plan](../dev_docs/phase_1/refining_config_handling.md)  
- **Expected Outcome**: Reduced errors, improved configuration handling.  
📅 **Timeline**: Week 2-3: Refactor tools.

### **3. 🔄 Reprocessing and Selective Workflow Execution**  
![Effort](https://img.shields.io/badge/effort-MEDIUM-orange) ![Category](https://img.shields.io/badge/category-Flexibility-yellow)  
**Description**: Allow users to reprocess specific images or steps of the pipeline without rerunning the entire workflow.  
- **Action Items**:
  - Develop a tool that enables step-based reprocessing.
  - Implement metadata restoration capabilities.  
- **Development Plan**: [View Development Plan](../dev_docs/phase_1/reprocessing_workflow.md)  
- **Expected Outcome**: Time and resource savings by focusing only on necessary reprocessing.  
📅 **Timeline**: Week 4-5: Tool development and testing.

---

## **Phase 2: Core Features & Cloud Integration**  
**Goal**: Optimize workflows, integrate cloud-based functionalities for scalability, and enhance flexibility for users.

### **1. ☁️ Cloud Integration for Metadata Management and Reprocessing**  
![Effort](https://img.shields.io/badge/effort-HIGH-red) ![Category](https://img.shields.io/badge/category-Scalable-green)  
**Description**: Migrate metadata management to cloud services (e.g., AWS RDS) for remote storage and processing, enabling larger datasets and automated workflows.  
- **Action Items**:
  - Set up cloud-based metadata storage and integrate with S3 triggers.
  - Automate reprocessing using AWS Lambda functions.  
- **Development Plan**: [View Development Plan](../dev_docs/phase_2/cloud_integration.md)  
- **Expected Outcome**: Scalable, automated workflows in the cloud.  
📅 **Timeline**: Month 2-3: Cloud integration setup.

### **2. 🌐 Web-Based OID + Image Preview Viewer**  
![Effort](https://img.shields.io/badge/effort-MEDIUM-orange) ![Category](https://img.shields.io/badge/category-Flexibility-yellow)  
**Description**: Develop a **web-based viewer** to preview **Oriented Imagery Datasets (OID)** and images, allowing easy access for remote users.  
- **Action Items**:
  - Build a web viewer using **React** or **Flask** for metadata display and image manipulation.  
- **Development Plan**: [View Development Plan](../dev_docs/phase_2/oid_image_viewer.md)  
- **Expected Outcome**: Flexible image and metadata access.  
📅 **Timeline**: Month 3: Web viewer development.

### **3. 💻 CLI and GUI Config Editors**  
![Effort](https://img.shields.io/badge/effort-MEDIUM-orange) ![Category](https://img.shields.io/badge/category-Flexibility-yellow)  
**Description**: Provide a **CLI runner** for independent pipelines and develop a **GUI config editor** for users to modify settings with ease.  
- **Action Items**:
  - Implement the **CLI runner** to execute workflows via the command line.
  - Develop a **GUI editor** (e.g., Electron or Jupyter-based) for easy configuration management.  
- **Development Plan**: [View Development Plan](../dev_docs/phase_2/cli_gui_config.md)  
- **Expected Outcome**: More flexible tool access.  
📅 **Timeline**: Month 3-4: CLI/GUI tool development.

---

## **Phase 3: Future-Proofing, Automation, and Innovation**  
**Goal**: Establish long-term capabilities like cloud-native operations, automation, and innovative features for enhanced performance and flexibility.

### **1. 🚀 Fully Automated Cloud-Based Pipeline**  
![Effort](https://img.shields.io/badge/effort-HIGH-red) ![Category](https://img.shields.io/badge/category-Scalable-green)  
**Description**: Transition the image processing pipeline to **cloud-native services** (e.g., AWS Fargate, ECS), fully automating each step for scalability and efficiency.  
- **Action Items**:
  - Containerize tools for deployment on AWS services.
  - Use **AWS Step Functions** for orchestration of the full pipeline.  
- **Development Plan**: [View Development Plan](../dev_docs/phase_3/fully_automated_pipeline.md)  
- **Expected Outcome**: Scalable, automated processing pipeline.  
📅 **Timeline**: Month 4-5: Full cloud-native transition.

### **2. 🤖 AI-Based Image Enhancement**  
![Effort](https://img.shields.io/badge/effort-HIGH-red) ![Category](https://img.shields.io/badge/category-Innovation-purple)  
**Description**: Integrate **AI-driven image enhancement** (e.g., blur detection, automatic contrast adjustments) into the pipeline to improve output quality.  
- **Action Items**:
  - Research and implement AI-based models for enhancing or processing images based on their content.  
- **Development Plan**: [View Development Plan](../dev_docs/phase_3/ai_image_enhancement.md)  
- **Expected Outcome**: Improved image quality through intelligent processing.  
📅 **Timeline**: Month 6: Research and model integration.

### **3. 📤 Cloud-Based Image Syncing and Publishing**  
![Effort](https://img.shields.io/badge/effort-MEDIUM-orange) ![Category](https://img.shields.io/badge/category-Scalable-green)  
**Description**: Automate the **uploading and syncing** of images to cloud storage (e.g., S3) with integrated metadata publishing to external viewers (e.g., ArcGIS Online).  
- **Action Items**:
  - Implement **S3 triggers** for image and metadata syncing.
  - Develop APIs for clients to upload data directly into the pipeline.  
- **Development Plan**: [View Development Plan](../dev_docs/phase_3/cloud_based_image_syncing.md)  
- **Expected Outcome**: Streamlined client workflows and data access.  
📅 **Timeline**: Month 6-7: Cloud syncing implementation.

---

## **Phase 4: Long-Term Growth & User Experience Enhancements**  
**Goal**: Focus on improving user experience, developing seamless workflows, and expanding the scope of features available to users.

### **1. 👁️ Standalone Viewer for Web and External Access**  
![Effort](https://img.shields.io/badge/effort-MEDIUM-orange) ![Category](https://img.shields.io/badge/category-Nice--to--Have-lightgrey)  
**Description**: Develop a **standalone image viewer** (e.g., **Photo Sphere Viewer** or **WebGL**) to access images directly from cloud storage.  
- **Action Items**:
  - Create a standalone viewer for users to view images and metadata from the cloud.  
- **Development Plan**: [View Development Plan](../dev_docs/phase_4/standalone_viewer.md)  
- **Expected Outcome**: Simplified access to images for users outside of ArcGIS.  
📅 **Timeline**: Month 8: Standalone viewer development.

### **2. 📈 Advanced Reporting and Analytics**  
![Effort](https://img.shields.io/badge/effort-MEDIUM-orange) ![Category](https://img.shields.io/badge/category-Refinement-orange)  
**Description**: Generate **automated reports** based on the processed images, summarizing key metrics, errors, and quality checks.  
- **Action Items**:
  - Integrate reporting tools to automatically generate PDF/HTML reports summarizing processing results.  
- **Development Plan**: [View Development Plan](../dev_docs/phase_4/advanced_reporting.md)  
- **Expected Outcome**: Clear reporting for project management.  
📅 **Timeline**: Month 8-9: Reporting implementation.

### **3. 📊 AI-Powered Cloud Analytics**  
![Effort](https://img.shields.io/badge/effort-HIGH-red) ![Category](https://img.shields.io/badge/category-Innovation-purple)  
**Description**: Implement **AI-driven analytics** for advanced image analysis, metadata extraction, and anomaly detection.  
- **Action Items**:
  - Research and deploy AI models for extracting more detailed insights from images and metadata.  
- **Development Plan**: [View Development Plan](../dev_docs/phase_4/ai_powered_analytics.md)  
- **Expected Outcome**: Intelligent image analysis for improved decision-making.  
📅 **Timeline**: Month 9-10: AI analytics deployment.

---

## **Next Steps**  
1. **Begin Phase 1 Development**: Focus on **Centralizing Metadata Management** and **Refining Config Handling**.
2. **Initiate Cloud Integration** for future scalability and automate metadata syncing.
3. **Prepare for Phase 2** by experimenting with **Web-Based Viewers** and building a **CLI Runner**.
4. **Begin AI Research** for **Image Enhancement** and **Cloud Analytics** in Phase 3.
