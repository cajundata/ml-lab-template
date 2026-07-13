**The only required probe whose success depends on something outside DigitalOcean.**

`transformers_smoke` pulls `facebook/opt-125m` from the Hub at run time. So a Hub outage, a rate limit, or a droplet with restricted egress fails a **required** probe — and therefore fails `make gpu-run` — on hardware that is working perfectly.

No local cache, no mirror, no retry. The model is small (125M), so the fix is cheap when it becomes worth doing: bake it into the image, or pre-fetch it during cloud-init where a failure is at least visible in `bootstrap.log`.
