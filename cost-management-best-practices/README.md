# Cost Management Best Practices

This document outlines some best practices to manage costs.

## Native Domino Functionality 

Domino offers two major levers to save on costs:

1. Auto-Shutdown of Long Running Workspaces
2. Auto-Deletion of Stopped Workspaces
3. Size HW Tiers appropriately


###  Auto-Shutdown of Long Running Workspaces

The most effective way to control costs is to stop your workspaces when they are not being used. Domino offers a mechanism
to configure auto-shutdown of workspaces when a fixed pre-defined interval of time has passed after the workspace starts.

The following links outline how to configure these settings:

1. [Domino Administrators](https://docs.dominodatalab.com/en/latest/user_guide/815d95/configure-long-running-workspaces/)
   configure workspaces as [long running workspaces](https://docs.dominodatalab.com/en/latest/admin_guide/71d6ad/central-configuration/#long-running)
    
2. Scroll down a bit on the second [link]((https://docs.dominodatalab.com/en/latest/admin_guide/71d6ad/central-configuration/#long-running)) and it shows how to configure the auto-shutdown

> **Warning**: The workspace will shutdown one the interval has passed even if you are actively working on it. The interval is
> the time elapsed after the workspace starts and not **idle time**.

> **Note** : This setting does not affect Jobs. A Job will not shutdown while it is running based on this setting

### Auto Deletion of Stopped Workspaces 

Another silent source of cost are EBS volumes attached to workspaces. Domino allows users to stop and start workspaces.
Domino maintains the state of the stopped workspace in EBS volumes (and snapshots). Shutting down a long running workspace
saves on compute costs. But the storage cost continues to be incurred until the workspace is deleted. 

It is not uncommon for users to not delete a workspace, especially for the projects they are not currently working on. 
Such stopped workspaces incurs the cost of storage. And as the number of users increase, this cost adds up significantly.

Domino allows your administrators to configure a setting, which will delete workspaces if they are not restarted for a fixed
amount of time (Default is 30 days). The user will be alerted (in the Domino Notifications Panel) as well as email that a 
workspace is going to be deleted X number of days (configurable) before the workspace is deleted.

This [link](https://docs.dominodatalab.com/en/latest/admin_guide/71d6ad/central-configuration/#workspaces) outlines 
the various levers you can avail of to save on storage costs.

### Size HW Tiers Appropriately

Assume that you use the same underlying node-pool for `Small`, `Medium` and `Large` instances. Assume you are on EKS 
and the instance type associated with the node-pool is `m5.4xlarge` which has 16 vCPU amd 64 GB memory.

Now assume that you define your tiers are follows

| HW Tiers    | Cores | Memory
| ----------- | ----------- | ----------- |
| Small      | 2       | 16 GB
| Medium   | 5        | 32 GB
| Large   | 10        | 60 GB

If all your users start up workspaces with the Large HW Tier, you will be wasting approximately 5 cores per instance

Worse still if your users start a workspace with a Medium HW Tier, you will end up wasting 10 cores and 30 GB of memory
because there is no room to start another Medium HW Tier based Domino Workload on the same instance. And even if users
are starting up Workloads using the small instance, you will be able to squeeze in at most one.

Admittedly, this is a crafted example. But stay vary of using a single nodepool across all instances. It will result in
idle capacity.

## Use Cloud GPU Tiers Judiciously

In most cases it is wasteful to start a workspace in the GPU HW Tier. Usually it is fiscally judicious to use the GPU
HW tier only for Jobs. Do you development on the CPU HW Tier and when you are ready launch a job in the GPU HW Tier

Examples of how to do so are illustrated in the [accompanying notebooks](execute_job.ipynb)

