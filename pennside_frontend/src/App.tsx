import { lazy, Suspense, useState } from "react";
import Landing from "./Landing";
const PostSignIn = lazy(() => import("./PostSignIn"));

export default function App() {
  const [signedIn, setSignedIn] = useState(false);
  const [userEmail, setUserEmail] = useState("");

  return signedIn ? (
    <Suspense fallback={<div className="app-loading">Preparing your PennSide…</div>}>
      <PostSignIn 
        email={userEmail}
        onSignOut={() => {
          setSignedIn(false);
          setUserEmail("");
        }} 
      />
    </Suspense>
  ) : (
    <Landing onSignedIn={(email: string) => {
      setUserEmail(email.trim().toLowerCase());
      setSignedIn(true);
    }} />
  );
}