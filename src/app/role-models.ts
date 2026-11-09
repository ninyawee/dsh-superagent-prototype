/** Resolves and applies per-role models on managed sessions without mutating the global DSH default. */
import type { Context } from '@deepseek-ai/cordis'
import type { Agent, ModelSelection } from '@deepseek-ai/dsh-agent'
import { ReasoningEffortId } from '@deepseek-ai/dsh-llm'
import type { RoleModel } from '../policy.ts'

/** Resolve a configured route through the live provider; catalog membership is not a restriction. */
export async function resolveRoleModel(ctx: Context, model: RoleModel): Promise<ModelSelection> {
  const value = await ctx.llm.resolveCallConfig({ provider: model.provider, model: model.model,
    ...(model.reasoningEffort ? { reasoningEffort: ReasoningEffortId(model.reasoningEffort) } : {}) })
  return { provider: value.provider, model: value.model, ...(value.reasoningEffort ? { reasoningEffort: value.reasoningEffort } : {}) }
}

/** Whether two role models name the same route and effort. */
export function sameRoleModel(left: RoleModel | null | undefined, right: RoleModel | null | undefined): boolean {
  return (left ?? null) === (right ?? null) || !!left && !!right && left.provider === right.provider && left.model === right.model
    && (left.reasoningEffort ?? null) === (right.reasoningEffort ?? null)
}

/**
 * Apply a role's configured model to a session. A brand-new session gets it before its first request; a session that has
 * already run keeps whatever it last used until the configured model differs from `applied`, the model this function last
 * applied to it, so a later manual selection survives restarts and only a settings change moves the session again.
 * DSH keeps an existing session on the route of its last request forever, so without this a session created on a provider
 * that later fails (such as one with no credential) keeps failing however the settings or the profile default change.
 * @returns the model now applied, to record as the next call's `applied`; `applied` unchanged when nothing was done.
 */
export async function applyRoleModel(ctx: Context, agent: Agent, model: RoleModel | null,
  applied: RoleModel | null = null): Promise<RoleModel | null> {
  if (!model) return applied
  const fresh = !agent.session.requestHeader() && !agent.session.snapshotEvents().some(event => event.type === 'model/selection')
  if (fresh) { await initializeRoleModel(ctx, agent, model); return model }
  if (sameRoleModel(model, applied)) return applied
  const selected = await resolveRoleModel(ctx, model)
  // A recorded selection becomes the session's next route; DSH adds the model-switch notice to the next step.
  agent.session.append('model/selection', selected)
  await ctx.sessionPersistence.flush()
  return model
}

/** Seed a role's configured model onto a brand-new session before its first request, leaving later manual or durable selections intact. */
export async function initializeRoleModel(ctx: Context, agent: Agent, model: RoleModel | null): Promise<void> {
  if (!model) return
  if (agent.session.requestHeader() || agent.session.snapshotEvents().some(event => event.type === 'model/selection')) return
  const selected = await resolveRoleModel(ctx, model)
  // Avoid selectModel because it mutates the global DSH default; inject the selection for the initial request only.
  agent.session.append('model/selection', selected)
  let initial = true
  let assembled: ModelSelection | undefined
  agent.ctx.effect(() => {
    const disposers = [
      agent.ctx.on('system-prompt/assemble', async (_assembly, _context, next) => {
        const selection = initial ? selected : undefined
        const result = await next()
        assembled = selection
        return selection ? { ...result, variables: { ...result.variables, provider: selection.provider, model: selection.model } } : result
      }, { prepend: true }),
      agent.ctx.on('agent/request', async (_payload, next) => {
        const result = await next()
        if (!assembled) return result
        const { reasoningEffort: _inherited, ...rest } = result
        return { ...rest, ...assembled }
      }, { prepend: true }),
      ctx.on('session/event', (session, event) => {
        if (session === agent.session && (event.type === 'request/header' || event.type === 'model/selection')) initial = false
      }),
    ]
    return () => disposers.forEach(dispose => dispose())
  })
  await ctx.sessionPersistence.flush()
}
