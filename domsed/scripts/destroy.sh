#!/bin/bash
set -x
echo "USAGE: ./destroy.sh"

name="operator"
platform_namespace="${platform_namespace:-domino-platform}"
compute_namespace="${compute_namespace:-domino-compute}"

secret="${name}-webhook-certs"
service="${name}-webhook-svc"

kubectl delete mutatingwebhookconfiguration ${name}-webhook

kubectl delete secret ${secret} -n ${platform_namespace}

kubectl delete peerauthentication ${name}-webhook -n ${platform_namespace}

kubectl delete deployment -n ${platform_namespace} "${name}-webhook"
kubectl delete networkpolicy -n ${platform-namespace} "${name}-webhook"
kubectl delete rolebinding ${name}-webhook  -n ${platform_namespace}
kubectl delete role ${name}-webhook  -n ${platform_namespace}
kubectl delete rolebinding ${name}-webhook  -n ${compute_namespace}
kubectl delete role ${name}-webhook  -n ${compute_namespace}
kubectl delete serviceaccount ${name}-webhook  -n ${platform_namespace}

kubectl delete svc -n ${platform_namespace} ${service}

kubectl delete crd mutations.apps.dominodatalab.com

kubectl label namespace ${compute_namespace} "${name}-enabled"-
